"""Public stateful plotting API shared by notebooks, Qt5 and headless use."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from threading import Event, RLock, current_thread
from time import monotonic
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, TypeAlias, TypeVar
import math

import numpy as np

from zlc_data import (
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetSchema,
    DatasetComponentValidity,
    DatasetRevisionRef,
    GridTopology,
    IndexRangeSelection,
    IndexSelection,
    OwnedSnapshot,
    PointTable,
    PointColumn,
    REPEAT,
    SCALAR,
    SCAN_POINT,
    Selection,
    ValidityContract,
    Value,
    ValueSchema,
    materialize_derived_dataset,
    materialize_scalar_dataset,
)
from zlc_data.snapshot_projection import (
    materialize_dataset_acceptance_mask,
    materialize_dataset_selection,
)

from ._dataset_bridge import (
    RevisionError,
    _ReadonlyDataset as DatasetSnapshot,
    _UNITS as DEFAULT_UNITS,
    _resolve_unit as resolve_unit,
    bridge_snapshot,
)

from ._axis_transform import AxisTransform, canvas_physical_size
from ._pulse_time import pulse_time_scale
from ._fit_projection import (
    FitProjection,
    FitScope,
    FitSelection,
    ProjectionContext,
)
from ._fit_scene import FitOverlay
from ._selector_scene import (
    ColorLimitCandidate,
    SelectorScene,
    SelectorSceneKind,
)
from ._validation import readonly_copy as _readonly
from .config import DEFAULTS, PlotLibraryDefaults
from .fit import (
    FacetFitBatchResult,
    FitCancelled,
    FitEngine,
    FitModelSpec,
    FitOptions,
    FitParameterDisplay,
    FitResult,
)
from .kinds import AxisDomain, AxisRef, PlotKind
from .layout import FacetTopology, SurfacePlan, facet_image_cell_aspect, resolve_surface
from .parameters import ParameterSchema, RenderEffect
from .primitives import (
    ImageFrame,
    ImagePointOverlay,
    PlotInput,
    PulseAnalogTrace,
    PulseBlock,
    PulseChannel,
    PulseDacScanSegment,
    PulseRepeatMarker,
    PulseScanRegion,
    PulseTimelineData,
)
from .rendering import MatplotlibRenderer, RenderFrame
from .resolver import index_parameter_specs
from .selectors import (
    CrosshairPoint,
    DragHandle,
    NumericRange,
    RectangleRange,
    _SelectorController,
    SelectorKind,
    SelectorSnapshot,
    SelectorState,
    SelectorValue,
    _drag_numeric_range,
)
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSpec,
    PulseTimelinePlot,
    RollingPlot,
    parameter_schema_for,
)
from .state import DisplayState, DisplayStateStore


_ProjectionInput = DatasetSnapshot | PulseTimelineData
SurfaceCallback = Callable[[], object]
HostDispatch = Callable[[Callable[[], Any]], Future[Any]]
HostPresentationDispatch = Callable[
    [Callable[[], None], Callable[[], None], Callable[[], None]],
    Future[Any],
]
DisplayCallback = Callable[[DisplayState], object]
FitCallback = Callable[["FitEvent"], object]
SelectionCallback = Callable[["SelectionEvent"], object]
_CallbackT = TypeVar("_CallbackT", bound=Callable[..., object])
_EventT = TypeVar("_EventT")
_ResultT = TypeVar("_ResultT")


class SelectionChange(str, Enum):
    ADDED = "added"
    UPDATED = "updated"
    COMMITTED = "committed"
    REMOVED = "removed"


_FIT_THREAD_PREFIX = "zlc-fit"
_UNSET = object()


def _validated_device_pixel_ratio(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("device pixel ratio must be a positive finite number")
    selected = float(value)
    if not math.isfinite(selected) or selected <= 0.0:
        raise ValueError("device pixel ratio must be a positive finite number")
    return selected


def _range_endpoint_hit(
    value: NumericRange,
    coordinate: float,
    tolerance: float,
) -> tuple[float, DragHandle] | None:
    endpoints = (
        (abs(coordinate - value.low) / tolerance, DragHandle.LOW),
        (abs(coordinate - value.high) / tolerance, DragHandle.HIGH),
    )
    score, handle = min(endpoints, key=lambda item: item[0])
    return (score, handle) if score <= 1.0 else None


@dataclass(frozen=True, slots=True)
class SessionRevisions:
    data: int
    display: int
    layout: int


@dataclass(frozen=True, slots=True)
class DisplayDescription:
    """Immutable control-plane snapshot for a notebook or GUI frontend."""

    kind: PlotKind
    size: str
    size_choices: tuple[str, ...]
    parameter_schema: ParameterSchema
    display_state: DisplayState
    parameter_choices: Mapping[str, tuple[object, ...]]
    limits: RectangleRange
    viewport: RectangleRange | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PlotKind):
            raise TypeError("display description kind must be PlotKind")
        if not isinstance(self.size, str) or not self.size:
            raise ValueError("display description size must be non-empty")
        size_choices = tuple(self.size_choices)
        if self.size not in size_choices:
            raise ValueError("current display size must be one of size_choices")
        if not isinstance(self.parameter_schema, ParameterSchema):
            raise TypeError("display description requires ParameterSchema")
        if not isinstance(self.display_state, DisplayState):
            raise TypeError("display description requires DisplayState")
        parameter_choices = {
            str(name): tuple(values)
            for name, values in self.parameter_choices.items()
        }
        unknown = tuple(
            name for name in parameter_choices if name not in self.parameter_schema
        )
        if unknown:
            joined = ", ".join(repr(name) for name in unknown)
            raise KeyError(f"parameter choices refer to unknown parameter(s): {joined}")
        object.__setattr__(self, "size_choices", size_choices)
        object.__setattr__(
            self,
            "parameter_choices",
            MappingProxyType(parameter_choices),
        )
        if not isinstance(self.limits, RectangleRange):
            raise TypeError("display description limits must be RectangleRange")
        if self.viewport is not None and not isinstance(self.viewport, RectangleRange):
            raise TypeError("display description viewport must be RectangleRange or None")


@dataclass(frozen=True, slots=True)
class PlotSessionConfig:
    """The small state needed to recreate one plot session."""

    spec: PlotSpec
    size: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.size, str) or not self.size:
            raise ValueError("plot session size must be non-empty text")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("plot session parameters must be a mapping")
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )


@dataclass(frozen=True, slots=True)
class _ColorLimitDrag:
    """One canonical color-scale gesture, separate from data selectors."""

    original: NumericRange
    candidate: NumericRange
    handle: DragHandle
    origin: float
    bounds: NumericRange
    minimum_span: float

    @property
    def changed(self) -> bool:
        return not np.allclose(
            (self.candidate.low, self.candidate.high),
            (self.original.low, self.original.high),
            rtol=1.0e-12,
            atol=1.0e-15,
        )

    def moved(self, position: float) -> "_ColorLimitDrag":
        return replace(
            self,
            candidate=_drag_numeric_range(
                self.original,
                handle=self.handle,
                origin=self.origin,
                position=position,
                minimum_span=self.minimum_span,
                bounds=self.bounds,
            ),
        )


@dataclass(slots=True)
class _PointerGestureBase:
    axes: Any
    transform: AxisTransform
    _cadence_at: dict[str, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def lane_due(self, lane: str, interval_ms: int) -> bool:
        now = monotonic()
        previous = self._cadence_at.get(lane)
        if previous is not None and now - previous < float(interval_ms) / 1000.0:
            return False
        self._cadence_at[lane] = now
        return True


@dataclass(slots=True)
class _SelectorGesture(_PointerGestureBase):
    kind: SelectorKind
    external_scene: bool


@dataclass(slots=True)
class _ColorGesture(_PointerGestureBase):
    drag: _ColorLimitDrag
    external_scene: bool


@dataclass(slots=True)
class _PanGesture(_PointerGestureBase):
    origin: CrosshairPoint
    x: NumericRange
    y: NumericRange
    candidate: RectangleRange | None = None


_PointerGesture: TypeAlias = _SelectorGesture | _ColorGesture | _PanGesture


@dataclass(frozen=True, slots=True)
class SelectionData:
    selector: SelectorState
    selection: Selection | None
    selected_value: Value | None
    source_revisions: tuple[int, ...]
    data_revision: int
    facet_index: int | None = None
    _source: OwnedSnapshot | None = field(default=None, repr=False, compare=False)
    _accepted_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    _rolling_values: np.ndarray | None = field(default=None, repr=False, compare=False)
    _rolling_valid: np.ndarray | None = field(default=None, repr=False, compare=False)
    _rolling_unit: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.selector, SelectorState):
            raise TypeError("selector must be SelectorState")
        if self.selection is not None and not isinstance(self.selection, Selection):
            raise TypeError("selection must be zlc_data.Selection or None")
        if self.selected_value is not None and not isinstance(self.selected_value, Value):
            raise TypeError("selected_value must be zlc_data.Value or None")
        revisions = tuple(self.source_revisions)
        if not revisions or any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for value in revisions
        ):
            raise TypeError("source_revisions must contain non-negative integers")
        object.__setattr__(self, "source_revisions", tuple(map(int, revisions)))
        if self._source is not None and not isinstance(self._source, OwnedSnapshot):
            raise TypeError("selection source must be OwnedSnapshot")
        if self._accepted_mask is not None:
            accepted = np.asarray(self._accepted_mask, dtype=bool)
            if self._source is None or accepted.shape != self._source.block.schema.physical_shape:
                raise ValueError("selection acceptance mask disagrees with its source")
            accepted.setflags(write=False)
            object.__setattr__(self, "_accepted_mask", accepted)
        rolling_values = self._rolling_values
        rolling_valid = self._rolling_valid
        if rolling_values is None:
            if rolling_valid is not None or self._rolling_unit is not None:
                raise ValueError("rolling selection metadata requires rolling values")
        else:
            if self.selection is not None or self._accepted_mask is not None:
                raise ValueError("rolling selection cannot also carry a source mask")
            values = np.asarray(rolling_values).reshape(-1).copy()
            if values.dtype.kind not in "biuf":
                raise TypeError("rolling selection values must be real numeric")
            valid = np.asarray(rolling_valid, dtype=np.bool_).reshape(-1).copy()
            if values.size == 0 or valid.shape != values.shape:
                raise ValueError("rolling values and validity must be non-empty and aligned")
            if len(self.source_revisions) != values.size:
                raise ValueError(
                    "rolling values must have one exact source revision per sample"
                )
            values.setflags(write=False)
            valid.setflags(write=False)
            object.__setattr__(self, "_rolling_values", values)
            object.__setattr__(self, "_rolling_valid", valid)
            if self._rolling_unit is not None:
                unit = str(self._rolling_unit).strip()
                object.__setattr__(self, "_rolling_unit", unit or None)

    def materialize(self, *, reference_for: Callable[[Any], Any]) -> OwnedSnapshot:
        """Materialize this committed selection through zlc_data's sole owner."""

        if self._rolling_values is not None:
            if not callable(reference_for):
                raise TypeError("reference_for must be callable")
            values = self._rolling_values
            valid = self._rolling_valid
            assert valid is not None
            schema = DatasetSchema(
                AxisSpec(
                    AxisId("selection.repeat"),
                    "repeat",
                    REPEAT,
                    1,
                    (0,),
                ),
                PointTable(values.size),
                None,
                ValueSchema.scalar(values.dtype, self._rolling_unit),
            )
            ref = reference_for(schema)
            if ref.schema_fingerprint != schema.fingerprint:
                raise ValueError("derived reference schema differs from rolling selection")
            if ref.revision.value != self.data_revision:
                raise ValueError("derived rolling selection must retain source revision")
            if self._source is not None and ref.block_id == self._source.ref.block_id:
                raise ValueError("derived rolling selection requires a new BlockId")
            return OwnedSnapshot(
                ref,
                DataBlock(
                    ref.block_id,
                    ref.revision,
                    values.reshape(schema.physical_shape),
                    CellValidity(valid.reshape(1, values.size)),
                    schema,
                ),
            )
        if self._source is None:
            raise RuntimeError("selection has no Dataset source")
        if self.selection is not None:
            return materialize_dataset_selection(
                self._source,
                self.selection,
                reference_for=reference_for,
            )
        if self._accepted_mask is None:
            raise ValueError("this selector does not define a Dataset selection")
        return materialize_dataset_acceptance_mask(
            self._source,
            self._accepted_mask,
            reference_for=reference_for,
        )


@dataclass(frozen=True, slots=True)
class PulseTimelineSelectionData:
    """Immutable timeline records intersecting a selector's source-time span."""

    selector: SelectorState
    display_selector: SelectorState
    channels: tuple[PulseChannel, ...]
    blocks: tuple[PulseBlock, ...]
    analog_traces: tuple[PulseAnalogTrace, ...]
    scan_regions: tuple[PulseScanRegion, ...]
    scan_dac_segments: tuple[PulseDacScanSegment, ...]
    repeat_markers: tuple[PulseRepeatMarker, ...]
    data_revision: int

    @property
    def canonical_value(self) -> SelectorValue:
        return self.selector.value

    @property
    def display_value(self) -> SelectorValue:
        return self.display_selector.value

    def __post_init__(self) -> None:
        if isinstance(self.data_revision, bool) or not isinstance(
            self.data_revision, Integral
        ):
            raise TypeError("data_revision must be an integer")
        revision = int(self.data_revision)
        if revision < 0:
            raise ValueError("data_revision must be non-negative")
        record_fields = (
            ("channels", PulseChannel),
            ("blocks", PulseBlock),
            ("analog_traces", PulseAnalogTrace),
            ("scan_regions", PulseScanRegion),
            ("scan_dac_segments", PulseDacScanSegment),
            ("repeat_markers", PulseRepeatMarker),
        )
        for name, record_type in record_fields:
            records = tuple(getattr(self, name))
            if any(not isinstance(record, record_type) for record in records):
                raise TypeError(f"{name} must contain {record_type.__name__} values")
            object.__setattr__(self, name, records)
        object.__setattr__(self, "data_revision", revision)


SelectorData: TypeAlias = SelectionData | PulseTimelineSelectionData


@dataclass(frozen=True, slots=True)
class SelectionEvent:
    """One selector lifecycle event in canonical and display coordinates."""

    change: SelectionChange
    selector: SelectorState
    display_selector: SelectorState
    data_revision: int
    data: SelectorData | None = None

    def __post_init__(self) -> None:
        if self.change is SelectionChange.COMMITTED and self.data is None:
            raise ValueError("committed selection events require exact SelectionData")
        if self.data is not None and self.data.data_revision != self.data_revision:
            raise ValueError("selection event data revision mismatch")


def _fresh_fit_axis_id(
    base: str,
    used: set[AxisId],
) -> AxisId:
    suffix = 1
    while True:
        value = base if suffix == 1 else f"{base}-{suffix}"
        candidate = AxisId(value)
        if candidate not in used:
            return candidate
        suffix += 1


def _fit_scalar_schema(
    repeat_axis: AxisSpec,
    point_table: PointTable,
    topology: GridTopology | None,
    unit: str | None,
) -> DatasetSchema:
    return DatasetSchema(
        repeat_axis,
        point_table,
        topology,
        ValueSchema.scalar(np.dtype("<f8"), unit),
    )


def _facet_parameter_carrier(
    source: DatasetSchema,
    facet: AxisRef,
    facet_values: tuple[Any, ...],
    values: np.ndarray,
    valid: np.ndarray,
    unit: str | None,
) -> tuple[DatasetSchema, np.ndarray, Any]:
    """Map one FacetGrid result back onto its sole physical facet authority."""

    coordinates = tuple(facet_values)
    count = len(coordinates)
    if count == 0 or values.shape != (count,) or valid.shape != (count,):
        raise ValueError("facet parameter values must match the fitted cell order")
    used = {
        source.repeat_axis.axis_id,
        *(column.coordinate_id for column in source.point_table.columns),
        *(axis.axis_id for axis in source.cell_schema.data_axes),
    }
    if facet.domain is AxisDomain.REPEAT:
        axis = source.repeat_axis
        repeat = AxisSpec(
            axis.axis_id,
            axis.name,
            axis.role,
            count,
            coordinates,
            axis.unit,
            axis.coordinate_frame,
        )
        schema = _fit_scalar_schema(repeat, PointTable(1), None, unit)
        return (
            schema,
            values.reshape(schema.physical_shape),
            CellValidity(valid.reshape(count, 1)),
        )

    repeat_id = _fresh_fit_axis_id("zlc_plot.fit-result-repeat", used)
    repeat = AxisSpec(repeat_id, "repeat", REPEAT, 1, (0,))
    if facet.domain in {
        AxisDomain.POINT_ROW,
        AxisDomain.POINT_COORDINATE,
        AxisDomain.POINT_DIMENSION,
    }:
        if facet.domain is AxisDomain.POINT_ROW:
            point_id = _fresh_fit_axis_id(
                "zlc_plot.fit-result-point-row",
                used | {repeat_id},
            )
            column = PointColumn(
                point_id,
                "point",
                SCAN_POINT,
                PointColumn.NUMERIC,
                coordinates,
            )
        else:
            assert facet.axis_id is not None
            source_column = source.point_table.column(AxisId(facet.axis_id))
            column = PointColumn(
                source_column.coordinate_id,
                source_column.name,
                source_column.role,
                source_column.value_kind,
                coordinates,
                source_column.unit,
                source_column.coordinate_frame,
            )
        point_table = PointTable(count, (column,))
        topology = (
            GridTopology(
                (column.coordinate_id,),
                (coordinates,),
                tuple((index,) for index in range(count)),
            )
            if facet.domain is AxisDomain.POINT_DIMENSION
            else None
        )
        schema = _fit_scalar_schema(repeat, point_table, topology, unit)
        return (
            schema,
            values.reshape(schema.physical_shape),
            CellValidity(valid.reshape(1, count)),
        )

    if facet.domain is not AxisDomain.DATA or facet.axis_id is None:
        raise ValueError("unsupported FacetGrid parameter authority")
    source_axis = source.cell_schema.axis(AxisId(facet.axis_id))
    axis = AxisSpec(
        source_axis.axis_id,
        source_axis.name,
        source_axis.role,
        count,
        coordinates,
        source_axis.unit,
        source_axis.coordinate_frame,
    )
    if axis.role == SCALAR:
        if count != 1:
            raise ValueError("the scalar carrier cannot contain multiple facet cells")
        schema = _fit_scalar_schema(repeat, PointTable(1), None, unit)
        return (
            schema,
            values.reshape(schema.physical_shape),
            CellValidity(valid.reshape(1, 1)),
        )
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema(
            (axis,),
            ValidityContract.components(axis.axis_id),
            np.dtype("<f8"),
            unit,
        ),
    )
    return (
        schema,
        values.reshape(schema.physical_shape),
        DatasetComponentValidity(
            (axis.axis_id,),
            valid.reshape(schema.physical_shape),
        ),
    )


@dataclass(frozen=True, slots=True)
class FitEvent:
    """A fit result accepted and painted for the current data revision."""

    result: FitResult | FacetFitBatchResult
    selection: "FitSelection | tuple[FitSelection, ...]"
    display_parameters: tuple[FitParameterDisplay, ...]
    formula: str
    _source: OwnedSnapshot = field(repr=False, compare=False)
    _parameter_units: tuple[str | None, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self._source, OwnedSnapshot):
            raise TypeError("FitEvent source must be OwnedSnapshot")
        model = self.result.model
        units = tuple(self._parameter_units)
        if len(units) != len(model.parameters):
            raise ValueError("FitEvent parameter units disagree with its model")
        if any(unit is not None and (not isinstance(unit, str) or not unit) for unit in units):
            raise TypeError("FitEvent parameter units must be non-empty strings or None")
        source_revision = self._source.ref.revision.value
        result_revision = (
            self.result.source_revision
            if isinstance(self.result, FacetFitBatchResult)
            else self.result.data_revision
        )
        if source_revision != result_revision:
            raise ValueError("FitEvent source and result revisions differ")
        object.__setattr__(self, "_parameter_units", units)

    @property
    def source_revisions(self) -> tuple[int, ...]:
        """Ordered exact source revisions consumed by this fitted result."""

        selections = (
            self.selection
            if isinstance(self.selection, tuple)
            else (self.selection,)
        )
        ordered: list[int] = []
        for selection in selections:
            for revision in selection.source_revisions:
                if revision not in ordered:
                    ordered.append(revision)
        return tuple(ordered)

    def materialize_parameters(
        self,
        reference_for: Callable[[str, DatasetSchema], DatasetRevisionRef],
    ) -> dict[str, OwnedSnapshot]:
        """Publish named physical parameters without exposing fit curves."""

        if not callable(reference_for):
            raise TypeError("reference_for must be callable")
        if isinstance(self.result, FitResult):
            return self._materialize_single_parameters(reference_for)
        return self._materialize_facet_parameters(reference_for)

    def _materialize_single_parameters(
        self,
        reference_for: Callable[[str, DatasetSchema], DatasetRevisionRef],
    ) -> dict[str, OwnedSnapshot]:
        result = self.result
        assert isinstance(result, FitResult)
        output: dict[str, OwnedSnapshot] = {}
        identities: set[tuple[object, object]] = set()
        for parameter, value, unit in zip(
            result.model.parameters,
            result.parameter_values,
            self._parameter_units,
            strict=True,
        ):
            snapshot = materialize_scalar_dataset(
                self._source.ref,
                float(value),
                valid=bool(result.success and math.isfinite(float(value))),
                unit=unit,
                reference_for=lambda schema, name=parameter.name: reference_for(
                    name,
                    schema,
                ),
            )
            identity = (snapshot.ref.block_id, snapshot.ref.stream_generation)
            if identity in identities:
                raise ValueError("Fit parameters require distinct dataset identities")
            identities.add(identity)
            output[parameter.name] = snapshot
        return output

    def _materialize_facet_parameters(
        self,
        reference_for: Callable[[str, DatasetSchema], DatasetRevisionRef],
    ) -> dict[str, OwnedSnapshot]:
        result = self.result
        assert isinstance(result, FacetFitBatchResult)
        output: dict[str, OwnedSnapshot] = {}
        identities: set[tuple[object, object]] = set()
        for parameter_index, (parameter, unit) in enumerate(zip(
            result.model.parameters,
            self._parameter_units,
            strict=True,
        )):
            values = np.asarray(
                tuple(
                    np.nan
                    if cell is None
                    else float(cell.parameter_values[parameter_index])
                    for cell in result.results
                ),
                dtype=np.float64,
            )
            valid = np.asarray(
                tuple(
                    cell is not None
                    and cell.success
                    and math.isfinite(float(cell.parameter_values[parameter_index]))
                    for cell in result.results
                ),
                dtype=np.bool_,
            )
            schema, physical_values, validity = _facet_parameter_carrier(
                self._source.block.schema,
                result.facet,
                result.facet_values,
                values,
                valid,
                unit,
            )
            snapshot = materialize_derived_dataset(
                self._source.ref,
                physical_values,
                schema=schema,
                validity=validity,
                reference_for=lambda schema, name=parameter.name: reference_for(
                    name,
                    schema,
                ),
            )
            identity = (snapshot.ref.block_id, snapshot.ref.stream_generation)
            if identity in identities:
                raise ValueError("Fit parameters require distinct dataset identities")
            identities.add(identity)
            output[parameter.name] = snapshot
        return output


@dataclass(frozen=True, slots=True)
class _SelectionSubscription:
    callback: SelectionCallback
    selector_kind: SelectorKind | None


@dataclass(frozen=True, slots=True)
class _LiveFitRequest:
    model: FitModelSpec
    selector_kind: SelectorKind | None
    initial: Mapping[str, float] | tuple[float, ...] | None
    bounds: Mapping[str, tuple[float | None, float | None]] | None
    options: FitOptions | None


@dataclass(frozen=True, slots=True)
class _PointerUpdate:
    """One backend-neutral gesture reduction and its exact transient scene."""

    candidate: SelectorState | ColorLimitCandidate | None
    scene: SelectorScene | None
    role: str | None
    cell_index: int | None
    active_pan: bool
    publish_front: bool


@dataclass(frozen=True, slots=True)
class _StartedFitRequest:
    request: _LiveFitRequest
    selection: FitSelection
    cancellation: Event
    context_generation: int
    request_generation: int


@dataclass(frozen=True, slots=True)
class _StartedFacetFitRequest:
    request: _LiveFitRequest
    facet: AxisRef
    facet_values: tuple[Any, ...]
    selections: tuple[FitSelection, ...]
    cancellation: Event
    context_generation: int
    request_generation: int


@dataclass(frozen=True, slots=True)
class _ResolvedFit:
    """One complete solver result, selection authority and painted overlay."""

    result: FitResult
    selection: FitSelection
    overlay: FitOverlay

    def __post_init__(self) -> None:
        if not isinstance(self.result, FitResult):
            raise TypeError("resolved fit result must be FitResult")
        if not isinstance(self.selection, FitSelection):
            raise TypeError("resolved fit selection must be FitSelection")
        if not isinstance(self.overlay, FitOverlay):
            raise TypeError("resolved fit overlay must be FitOverlay")
        if self.result.data_revision != self.selection.data_revision:
            raise ValueError("fit result and selection revisions differ")
        if self.overlay.success != self.result.success:
            raise ValueError("fit result and overlay success differ")
        if self.overlay.facet_index != self.selection.facet_index:
            raise ValueError("fit selection and overlay facets differ")


@dataclass(frozen=True, slots=True)
class _FitResolution:
    completion: Future[FitResult]
    result: FitResult | None = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _AcceptedFit(_ResolvedFit):
    """One atomically accepted fit result and its painted presentation."""

    context_generation: int

    def __post_init__(self) -> None:
        _ResolvedFit.__post_init__(self)
        if isinstance(self.context_generation, bool) or not isinstance(
            self.context_generation, int
        ):
            raise TypeError("accepted fit context generation must be an integer")
        if self.context_generation < 0:
            raise ValueError("accepted fit context generation must be non-negative")


@dataclass(frozen=True, slots=True)
class _AcceptedFacetFit:
    result: FacetFitBatchResult
    selections: tuple[FitSelection, ...]
    overlays: tuple[FitOverlay, ...]
    context_generation: int

    def __post_init__(self) -> None:
        if len(self.selections) != len(self.result.facet_values):
            raise ValueError("facet fit selections disagree with batch cells")
        if any(not isinstance(item, FitSelection) for item in self.selections):
            raise TypeError("facet fit selections must contain FitSelection")
        if any(not isinstance(item, FitOverlay) for item in self.overlays):
            raise TypeError("facet fit overlays must contain FitOverlay")
        expected = tuple(
            selection.facet_index
            for selection, result in zip(
                self.selections,
                self.result.results,
                strict=True,
            )
            if result is not None
        )
        if tuple(item.facet_index for item in self.overlays) != expected:
            raise ValueError("facet fit overlays disagree with fitted cells")


@dataclass(frozen=True, slots=True)
class _ProjectionPresentation:
    """A drawn projection plus the exact state needed to abort publication."""

    committed_projection: FitProjection
    previous_projection: FitProjection
    previous_image_overlay: ImagePointOverlay | None
    previous_accepted_fit: _AcceptedFit | None
    previous_facet_thresholds: tuple[float | None, ...]
    previous_focused_facet_index: int | None
    previous_facet_focus_index: int | None
    previous_viewport: RectangleRange | None
    previous_layout_revision: int
    previous_plan: SurfacePlan
    surface_callbacks: tuple[SurfaceCallback, ...]


@dataclass(frozen=True, slots=True)
class _FitPresentation:
    event: FitEvent
    accepted: _AcceptedFit
    previous: _AcceptedFit | None
    previous_batch: _AcceptedFacetFit | None


@dataclass(frozen=True, slots=True)
class _FacetFitPresentation:
    event: FitEvent
    accepted: _AcceptedFacetFit
    previous_fit: _AcceptedFit | None
    previous_batch: _AcceptedFacetFit | None


class PlotSession:
    """One public plot surface with immutable input snapshots and public APIs.

    Data, display and layout revisions are independent. A fixed-size/layout
    edit is composed inside the existing Figure and promoted only after it has
    drawn successfully; the previously accepted front surface is
    untouched until then.
    """

    def __init__(
        self,
        data: PlotInput,
        spec: PlotSpec,
        *,
        size: str | None = None,
        parameters: Mapping[str, object] | None = None,
        defaults: PlotLibraryDefaults = DEFAULTS,
        device_pixel_ratio: float = 1.0,
        dispatch: HostDispatch | None = None,
        fit_engine: FitEngine | None = None,
    ) -> None:
        if not isinstance(defaults, PlotLibraryDefaults):
            raise TypeError("defaults must be PlotLibraryDefaults")
        data, initial_image_frame = self._split_image_frame(data, spec)
        FitProjection._validate_input(data, spec)
        self._lock = RLock()
        self._render_lock = RLock()
        self._ownership_gate = RLock()
        self._session_identity = object()
        self._closed = False
        self._defaults = defaults
        self._unit_registry = DEFAULT_UNITS
        self._spec = spec
        index_parameters, self._index_parameter_refs = (
            index_parameter_specs(data.schema, spec)
            if isinstance(data, DatasetSnapshot)
            else ((), {})
        )
        self._parameter_schema = parameter_schema_for(
            spec,
            style=defaults.style,
            extra_parameters=index_parameters,
        )
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        initial_parameters = {} if parameters is None else dict(parameters)
        self._display_store = DisplayStateStore(
            self._parameter_schema,
            initial_parameters,
        )
        self._size = (
            None if size is None else defaults.layout.validate_preset(size)
        )
        self._device_pixel_ratio = _validated_device_pixel_ratio(device_pixel_ratio)
        if dispatch is not None and not callable(dispatch):
            raise TypeError("dispatch must be callable or None")
        self._dispatch = dispatch
        self._presentation_dispatch: HostPresentationDispatch | None = None
        self._host_owner: object | None = None
        self._host_previous_dispatch: HostDispatch | None = None
        self._host_previous_presentation_dispatch: HostPresentationDispatch | None = None
        self._layout_revision = 0
        self._image_overlay = (
            None if initial_image_frame is None else initial_image_frame.overlay
        )
        self._renderer: MatplotlibRenderer | None = None
        self._surface_callbacks: list[SurfaceCallback] = []
        self._display_callbacks: list[DisplayCallback] = []
        self._fit_callbacks: list[FitCallback] = []
        self._selection_subscriptions: list[_SelectionSubscription] = []
        self._selector_controller = _SelectorController()
        if fit_engine is not None and not isinstance(fit_engine, FitEngine):
            raise TypeError("fit_engine must be FitEngine or None")
        self._fit_engine = fit_engine or FitEngine()
        self._accepted_fit: _AcceptedFit | None = None
        self._accepted_facet_fit: _AcceptedFacetFit | None = None
        self._facet_thresholds: tuple[float | None, ...] = ()
        self._fit_executor = ThreadPoolExecutor(
            max_workers=defaults.runtime.analysis_worker_count,
            thread_name_prefix=_FIT_THREAD_PREFIX,
        )
        self._fit_cancel = Event()
        self._fit_context_generation = 0
        self._fit_request_generation = 0
        self._live_fit_request: _LiveFitRequest | None = None
        self._live_fit_future: Future[FitResult] | None = None
        self._live_fit_completion: Future[FitResult] | None = None
        self._live_fit_pending = False
        self._viewport: RectangleRange | None = None
        self._focused_facet_index: int | None = (
            0 if isinstance(spec, FacetGridPlot) else None
        )
        self._facet_focus_index: int | None = None
        self._gesture: _PointerGesture | None = None
        self._click_history: dict[int, tuple[float, float, float]] = {}
        initial_revision = data.revision if isinstance(data, DatasetSnapshot) else 0
        self._projection = FitProjection(
            data=data,
            revision=initial_revision,
            spec=self._spec,
            context=self._projection_context(),
            defaults=self._defaults,
            histogram_projection=None,
        )
        self._rebuild_projection()
        self._presentation_epoch = 0
        plan = self._resolve_plan()
        # Automatic sizing is an initial recommendation.  Once consumed, the
        # resulting named preset is authoritative just like a user selection.
        self._size = plan.preset
        renderer = MatplotlibRenderer(spec, plan, style=defaults.style)
        self._update_renderer(renderer, RenderEffect.LAYOUT)
        self._renderer = renderer

    @staticmethod
    def _split_image_frame(
        data: PlotInput,
        spec: PlotSpec,
    ) -> tuple[_ProjectionInput, ImageFrame | None]:
        if isinstance(data, OwnedSnapshot):
            return bridge_snapshot(data), None
        if not isinstance(data, ImageFrame):
            return data, None
        if not isinstance(spec, ImagePlot):
            raise TypeError("ImageFrame requires ImagePlot")
        return bridge_snapshot(data.snapshot), data

    @staticmethod
    def _same_image_overlay(
        left: ImagePointOverlay,
        right: ImagePointOverlay,
    ) -> bool:
        return bool(
            left.revision == right.revision
            and left.point_ids == right.point_ids
            and left.labels == right.labels
            and left.statuses == right.statuses
            and np.array_equal(left.coordinates, right.coordinates)
        )

    @classmethod
    def _validate_image_frame_overlay(
        cls,
        previous: ImagePointOverlay | None,
        incoming: ImagePointOverlay,
    ) -> None:
        """Keep the point-layer revision monotonic across both update APIs."""

        if previous is None:
            return
        if incoming.revision < previous.revision:
            raise RevisionError(
                "ImageFrame overlay revision cannot move backwards"
            )
        if incoming.revision == previous.revision and not cls._same_image_overlay(
            previous,
            incoming,
        ):
            raise RevisionError(
                "one image overlay revision cannot identify different content"
            )

    def _projection_context(self) -> ProjectionContext:
        with self._lock:
            return ProjectionContext(
                display_state=self.display_state,
                selector_snapshot=self._selector_controller.snapshot(),
                viewport=self._viewport,
                focused_facet_index=self._focused_facet_index,
            )

    @property
    def _projected(self) -> FitProjection:
        """Return the current projection under one immutable session context."""

        return self._projection._with_context(self._projection_context())

    def _rebuild_projection(self, *, payload_only: bool = False) -> None:
        self._projection._reproject(
            context=self._projection_context(),
            payload_only=payload_only,
        )

    @property
    def _view(self) -> Any:
        return self._projection.view

    @property
    def _payload(self) -> Any:
        return self._projection.payload

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("plot session is closed")

    @property
    def spec(self) -> PlotSpec:
        return self._spec

    @property
    def configuration(self) -> PlotSessionConfig:
        """Immutable spec/display state for saved layouts and figures."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
                return PlotSessionConfig(
                    self._spec,
                    self.surface_plan.preset,
                    self.display_state.values,
                )

    def replace_spec(
        self,
        spec: PlotSpec,
        *,
        parameters: Mapping[str, object] | None = None,
    ) -> PlotSessionConfig:
        """Atomically change plot semantics on the existing Figure/canvas."""

        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping or None")
        completion: Future[FitResult] | None
        callbacks: tuple[SurfaceCallback, ...]
        with self._render_lock:
            with self._lock:
                self._assert_open()
                data = self._projection.data
                FitProjection._validate_input(data, spec)
                index_parameters, index_refs = (
                    index_parameter_specs(data.schema, spec)
                    if isinstance(data, DatasetSnapshot)
                    else ((), {})
                )
                parameter_schema = parameter_schema_for(
                    spec,
                    style=self._defaults.style,
                    extra_parameters=index_parameters,
                )
                if parameters is None:
                    old_values = self.display_state.values
                    initial = {
                        name: old_values[name]
                        for name in parameter_schema.names
                        if name in old_values
                    }
                else:
                    initial = dict(parameters)
                display_store = DisplayStateStore(
                    parameter_schema,
                    initial,
                    initial_revision=self.display_state.revision + 1,
                )
                focused = 0 if isinstance(spec, FacetGridPlot) else None
                projection = FitProjection(
                    data=data,
                    revision=self.data_revision,
                    spec=spec,
                    context=ProjectionContext(
                        display_store.state,
                        SelectorSnapshot(()),
                        None,
                        focused,
                    ),
                    defaults=self._defaults,
                    histogram_projection=None,
                )
                projection._build_view_and_payload()
                assert self._renderer is not None
                renderer = self._renderer
                old_plan = renderer.plan
                previous = (
                    self._spec,
                    self._parameter_schema,
                    self._index_parameter_refs,
                    self._display_store,
                    self._projection,
                    self._selector_controller,
                    self._image_overlay,
                    self._viewport,
                    self._focused_facet_index,
                    self._facet_focus_index,
                    self._accepted_fit,
                    self._accepted_facet_fit,
                    self._facet_thresholds,
                    self._layout_revision,
                )
                self._cancel_gesture()
                self._spec = spec
                self._parameter_schema = parameter_schema
                self._index_parameter_refs = index_refs
                self._display_store = display_store
                self._projection = projection
                self._selector_controller = _SelectorController()
                self._image_overlay = (
                    self._image_overlay if isinstance(spec, ImagePlot) else None
                )
                self._viewport = None
                self._focused_facet_index = focused
                self._facet_focus_index = None
                self._accepted_fit = None
                self._accepted_facet_fit = None
                self._facet_thresholds = ()
                self._layout_revision += 1
                renderer.spec = spec
                plan = self._resolve_plan()
            try:
                renderer.relayout(
                    plan,
                    facet_index=self._focused_facet_index,
                    facet_focus_index=None,
                )
                self._update_renderer(renderer, RenderEffect.LAYOUT)
            except BaseException:
                with self._lock:
                    (
                        self._spec,
                        self._parameter_schema,
                        self._index_parameter_refs,
                        self._display_store,
                        self._projection,
                        self._selector_controller,
                        self._image_overlay,
                        self._viewport,
                        self._focused_facet_index,
                        self._facet_focus_index,
                        self._accepted_fit,
                        self._accepted_facet_fit,
                        self._facet_thresholds,
                        self._layout_revision,
                    ) = previous
                    renderer.spec = self._spec
                renderer.relayout(
                    old_plan,
                    facet_index=self._focused_facet_index,
                    facet_focus_index=self._facet_focus_index,
                )
                self._update_renderer(renderer, RenderEffect.LAYOUT)
                raise
            with self._lock:
                self._fit_context_generation += 1
                self._fit_request_generation += 1
                self._fit_cancel.set()
                completion = self._live_fit_completion
                self._live_fit_completion = None
                self._live_fit_request = None
                self._live_fit_future = None
                self._live_fit_pending = False
                callbacks = tuple(self._surface_callbacks)
                config = PlotSessionConfig(
                    self._spec,
                    renderer.plan.preset,
                    self.display_state.values,
                )
        if completion is not None and not completion.done():
            completion.set_exception(FitCancelled("plot specification replaced"))
        self._notify_surface_callbacks(callbacks)
        self._notify_display(self.display_state)
        return config

    @property
    def defaults(self) -> PlotLibraryDefaults:
        """Immutable configuration used by this session and its frontends."""

        return self._defaults

    @property
    def surface_plan(self) -> SurfacePlan:
        with self._render_lock:
            assert self._renderer is not None
            return self._renderer.plan

    def _backend_surface(self) -> tuple[Any, SurfacePlan]:
        """Hand the private mutable surface only to an in-package canvas host."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            return self._renderer.figure, self._renderer.plan

    def _raster_axes_snapshot(
        self,
    ) -> tuple[AxisTransform, ...]:
        """Return immutable axis geometry without exposing Matplotlib objects."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            return tuple(
                self._axis_transform_for_axis(axis)
                for axis in self._renderer.figure.axes
                if bool(axis.get_visible())
            )

    def _raster_source_revisions_snapshot(self) -> tuple[int, ...]:
        """Return the exact source revisions represented by the painted front."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            if isinstance(self._spec, RollingPlot):
                revisions = tuple(getattr(self._payload, "source_revisions", ()))
                if not revisions:
                    raise RuntimeError("rolling raster has no represented source revisions")
                return tuple(map(int, revisions))
            return (self.data_revision,)

    def _canonical_axes_limits(
        self,
        role: str,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if role == "distribution":
            values = tuple(
                self._projected._display_scalar_to_canonical(
                    value,
                    self._projected._value_quantity(),
                )
                if self._view is not None
                else float(value)
                for value in y_limits
            )
            return values, (0.0, 0.0)
        if self._view is None:
            return tuple(map(float, x_limits)), tuple(map(float, y_limits))
        x_values = tuple(
            self._display_x_scalar_to_canonical(value) for value in x_limits
        )
        y_values = (
            tuple(map(float, y_limits))
            if self._projected._is_histogram_plot()
            else tuple(
                self._projected._display_scalar_to_canonical(
                    value,
                    self._projected._y_ref_or_value(),
                )
                for value in y_limits
            )
        )
        return x_values, y_values

    def _axis_transform_for_axis(self, axis: Any) -> AxisTransform:
        assert self._renderer is not None
        width, height = canvas_physical_size(self._renderer.figure.canvas)
        bbox = axis.get_window_extent()
        role, separator, suffix = str(axis.get_gid() or "main").partition(":")
        cell_index = int(suffix) if separator and suffix.isdigit() else None
        display_x = tuple(map(float, axis.get_xlim()))
        display_y = tuple(map(float, axis.get_ylim()))
        canonical_x, canonical_y = self._canonical_axes_limits(
            role or "main",
            display_x,
            display_y,
        )
        return AxisTransform(
            role or "main",
            cell_index,
            (
                float(bbox.x0) / width,
                1.0 - float(bbox.y1) / height,
                float(bbox.x1) / width,
                1.0 - float(bbox.y0) / height,
            ),
            display_x,
            display_y,
            canonical_x,
            canonical_y,
        )

    def _axis_for_transform(self, transform: AxisTransform) -> Any | None:
        assert self._renderer is not None
        for axis in self._renderer.axes.get(transform.role, ()):
            gid = str(axis.get_gid() or "")
            _role, separator, suffix = gid.partition(":")
            cell_index = int(suffix) if separator and suffix.isdigit() else None
            if cell_index == transform.cell_index:
                return axis
        return None

    def _selector_axes(self, state: SelectorState) -> Any | None:
        """Resolve backend axes at the session/renderer boundary."""

        assert self._renderer is not None
        if isinstance(self._spec, FacetGridPlot):
            axes = self._renderer.axes.get("facet_cell", ())
            index = (
                self._focused_facet_index
                if state.facet_index is None
                else state.facet_index
            )
            if index is None or index < 0 or index >= len(axes):
                return None
            return axes[index]
        return self._renderer.primary_axes

    def _raster_capture_rgba(
        self,
        *,
        redraw: bool = False,
    ) -> np.ndarray:
        """Capture the worker-owned, already-composed canvas."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            return self._renderer.capture_rgba(redraw=redraw)

    def _raster_interaction_snapshot(
        self,
    ) -> tuple[SelectorState, ...]:
        """Return the exact display-space selector state painted by this session."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            if (
                isinstance(self._spec, FacetGridPlot)
                and self._facet_focus_index is None
            ):
                return ()
            snapshot = self._display_selector_snapshot()
            gesture = self._gesture
            if not isinstance(gesture, (_SelectorGesture, _ColorGesture)):
                return snapshot.states
            if not gesture.external_scene:
                return snapshot.states
            if isinstance(gesture, _ColorGesture):
                return snapshot.committed
            return tuple(
                state
                for state in snapshot.committed
                if state.kind is not gesture.kind
            )

    def _raster_color_limits_snapshot(self) -> NumericRange | None:
        """Return the effective display-space clim painted into a raster front."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            if not self._renderer.axes.get("distribution"):
                return None
            try:
                low, high = self._renderer.resolved_color_limits()
            except TypeError:
                return None
            return NumericRange(*sorted((float(low), float(high))))

    def redraw_surface(self) -> None:
        """Rebuild the current canvas front after a host canvas is attached."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                self._renderer.draw()

    @property
    def parameter_schema(self) -> ParameterSchema:
        return self._parameter_schema

    @property
    def display_state(self) -> DisplayState:
        return self._display_store.state

    def _unit_parameter_sources(self) -> Mapping[str, Any]:
        if self._view is None:
            return MappingProxyType({})
        semantic = self._projected._semantic_spec()
        sources: dict[str, Any] = {
            "value_display_unit": self._projected._value_quantity(),
        }
        x_ref = getattr(semantic, "x", None)
        y_ref = getattr(semantic, "y", None)
        if isinstance(x_ref, AxisRef):
            sources["x_display_unit"] = self._projected._coordinate(x_ref)
        if isinstance(y_ref, AxisRef):
            sources["y_display_unit"] = self._projected._coordinate(y_ref)
        if isinstance(self._spec, FacetGridPlot):
            sources["facet_display_unit"] = self._projected._coordinate(
                self._spec.facet
            )
        return MappingProxyType(sources)

    def _parameter_choice_overrides(self) -> Mapping[str, tuple[object, ...]]:
        result: dict[str, tuple[object, ...]] = {}
        if self._view is None:
            if "x_display_unit" in self._parameter_schema:
                choices = self._parameter_schema["x_display_unit"].choices
                result["x_display_unit"] = tuple(map(str, choices))
            return MappingProxyType(result)
        registry = self._unit_registry or DEFAULT_UNITS
        for name, source in self._unit_parameter_sources().items():
            if name not in self._parameter_schema:
                continue
            compatible = []
            for symbol in registry.symbols():
                target = resolve_unit(symbol, registry)
                if source.canonical_unit.compatible_with(target):
                    compatible.append(symbol)
            result[name] = tuple(compatible)
        return MappingProxyType(result)

    def _current_display_limits(self) -> RectangleRange:
        assert self._renderer is not None
        axis = self._renderer.primary_axes
        x_low, x_high = sorted(map(float, axis.get_xlim()))
        y_low, y_high = sorted(map(float, axis.get_ylim()))
        return RectangleRange(
            self._viewport_x_from_axes(NumericRange(x_low, x_high)),
            NumericRange(y_low, y_high),
        )

    def describe_display(self) -> DisplayDescription:
        """Return one complete immutable snapshot for external controls."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            return DisplayDescription(
                kind=self._spec.kind,
                size=self.surface_plan.preset,
                size_choices=self._defaults.layout.size_names,
                parameter_schema=self._parameter_schema,
                display_state=self.display_state,
                parameter_choices=self._parameter_choice_overrides(),
                limits=self._current_display_limits(),
                viewport=self._viewport,
            )


    @property
    def selectors(self) -> tuple[SelectorState, ...]:
        """Return the effective immutable selector front."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            return self._resolved_selector_snapshot().states

    @property
    def facet_focus_index(self) -> int | None:
        """Presented FacetGrid cell, or ``None`` while showing the overview."""

        with self._lock:
            if not isinstance(self._spec, FacetGridPlot):
                return None
            return self._facet_focus_index

    def cancel_interaction(self) -> SelectorState | None:
        """Discard the transient selector candidate and release pointer state."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            return self._cancel_gesture()

    @property
    def last_fit(self) -> FitResult | None:
        with self._lock:
            return None if self._accepted_fit is None else self._accepted_fit.result

    @property
    def last_facet_fit(self) -> FacetFitBatchResult | None:
        """Latest batch result for the current immutable data revision."""

        with self._lock:
            accepted = self._accepted_facet_fit
            if (
                accepted is None
                or accepted.result.source_revision != self.data_revision
            ):
                return None
            return accepted.result

    @property
    def fit_status(self) -> str | None:
        """Whether the painted fit belongs to the current data revision.

        Selector/viewport/layout gestures are display operations.  They do not
        create a second fit clock or a transient ``lagging`` state; a fit is
        either the accepted result for this data frame or absent.
        """

        with self._lock:
            accepted = self._accepted_fit
            if accepted is None:
                return None
            current = (
                accepted.result.data_revision == self.data_revision
            )
        return "current" if current else None

    @property
    def live_fit_enabled(self) -> bool:
        """Whether accepted data/selector revisions automatically refit."""

        with self._lock:
            return self._live_fit_request is not None

    @property
    def fit_models(self) -> tuple[FitModelSpec, ...]:
        """Semantically and dimensionally valid models for the painted plot."""

        with self._render_lock:
            target = self._projected._fit_target()
            if target is None:
                return ()
            return tuple(
                model
                for model in self._fit_engine.registry.models_for(target)
                if self._projected._fit_model_units_compatible(model)
            )

    def _resolve_fit_model(self, model: str | FitModelSpec) -> FitModelSpec:
        resolved = self._fit_engine.registry.get(model) if isinstance(model, str) else model
        if not isinstance(resolved, FitModelSpec):
            raise TypeError("model must be a registered model id or FitModelSpec")
        return resolved

    @property
    def data_revision(self) -> int:
        with self._lock:
            return self._projection.data_revision

    @property
    def data_generation(self) -> str | None:
        """Dataset generation behind the current frame, if the kind uses one."""

        with self._lock:
            data = self._projection.data
            return str(data.generation) if isinstance(data, DatasetSnapshot) else None

    @property
    def revisions(self) -> SessionRevisions:
        with self._lock:
            return SessionRevisions(
                data=self.data_revision,
                display=self.display_state.revision,
                layout=self._layout_revision,
            )


    def _resolve_plan(self) -> SurfacePlan:
        topology = None
        if isinstance(self._spec, FacetGridPlot):
            cell_count = len(tuple(getattr(self._payload, "cells", ())))
            topology = FacetTopology(
                cell_count=max(cell_count, 1),
                cell_aspect=(
                    self._facet_image_aspect()
                    if isinstance(self._spec.cell, ImagePlot)
                    else None
                ),
            )
        side_distribution = None
        if isinstance(self._spec, RollingPlot):
            side_distribution = bool(self.display_state["side_distribution"])
        return resolve_surface(
            self._size,
            self._spec.kind,
            topology,
            device_pixel_ratio=self._device_pixel_ratio,
            rolling_side_distribution=side_distribution,
            layout=self._defaults.layout,
            style=self._defaults.style,
        )

    def _facet_image_aspect(self) -> float:
        """Return the first IMAGE cell's authored x/y domain aspect."""

        cells = tuple(getattr(self._payload, "cells", ()))
        if not cells:
            return 1.0
        payload = getattr(cells[0], "payload", None)
        x = np.asarray(getattr(getattr(payload, "x", None), "display", ())).reshape(-1)
        y = np.asarray(getattr(getattr(payload, "y", None), "display", ())).reshape(-1)
        if x.size == 0 or y.size == 0:
            return 1.0
        return facet_image_cell_aspect(int(x.size), int(y.size))

    def _update_renderer(
        self,
        renderer: MatplotlibRenderer,
        effects: RenderEffect,
    ) -> None:
        gesture = self._gesture
        viewport = (
            gesture.candidate
            if isinstance(gesture, _PanGesture) and gesture.candidate is not None
            else self._projected.viewport
        )
        view_limits = None
        if viewport is not None:
            axes_x = self._viewport_x_to_axes(viewport.x)
            view_limits = (
                (axes_x.low, axes_x.high),
                self._viewport_y_to_axes(viewport.y),
            )
        batch = self._accepted_facet_fit
        fit_overlay: FitOverlay | tuple[FitOverlay, ...] | None
        if (
            batch is not None
            and batch.result.source_revision == self.data_revision
            and isinstance(self._spec, FacetGridPlot)
        ):
            fit_overlay = batch.overlays
        else:
            fit_overlay = (
                None
                if (
                    self._accepted_fit is None
                    or self._accepted_fit.result.data_revision != self.data_revision
                )
                else self._accepted_fit.overlay
            )
        renderer.present(RenderFrame(
            payload=self._payload,
            state=self.display_state,
            effects=effects,
            data_revision=self.data_revision,
            fit_overlay=fit_overlay,
            facet_thresholds=tuple(
                None
                if value is None
                else self._projected._canonical_scalar_to_display(
                    value,
                    self._projected._value_quantity(),
                )
                for value in self._facet_thresholds
            ),
            image_overlay=self._image_overlay,
            selectors=self._display_selector_snapshot(),
            facet_index=self._focused_facet_index,
            facet_focus_index=self._facet_focus_index,
            view_limits=view_limits,
        ))
        self._presentation_epoch += 1

    def _render_current(
        self,
        effects: RenderEffect,
        *,
        schedule_fit: bool = False,
    ) -> None:
        with self._render_lock:
            assert self._renderer is not None
            self._update_renderer(self._renderer, effects)

    def _present_projection_transaction(
        self,
        projection: FitProjection,
        *,
        image_overlay: ImagePointOverlay | None,
        accepted_fit: _AcceptedFit | None,
    ) -> _ProjectionPresentation:
        """Swap one complete projected frame, restoring the old frame on failure."""

        if not isinstance(projection, FitProjection):
            raise TypeError("projection must be FitProjection")
        old_plan = self.surface_plan
        old_count = (
            old_plan.facet_topology.cell_count
            if isinstance(self._spec, FacetGridPlot)
            else None
        )
        new_count = (
            len(tuple(getattr(projection.payload, "cells", ())))
            if isinstance(self._spec, FacetGridPlot)
            else None
        )
        if old_count != new_count:
            self._cancel_gesture()
        with self._lock:
            previous = (
                self._projection,
                self._image_overlay,
                self._accepted_fit,
                self._facet_thresholds,
                self._focused_facet_index,
                self._facet_focus_index,
                self._viewport,
                self._layout_revision,
            )
            self._projection = projection
            self._image_overlay = image_overlay
            self._accepted_fit = accepted_fit
            self._facet_thresholds = ()
            if isinstance(self._spec, FacetGridPlot):
                assert new_count is not None
                self._clamp_facet_state(new_count)
                plan = self._resolve_plan() if old_count != new_count else None
            else:
                plan = None
            if plan is not None:
                self._layout_revision += 1
        try:
            if plan is not None:
                callbacks = self._apply_layout_plan(
                    plan,
                    schedule_fit=False,
                    notify_surface=False,
                )
            else:
                self._render_current(
                    RenderEffect.BASE_GEOMETRY,
                    schedule_fit=False,
                )
                callbacks = ()
        except BaseException:
            with self._lock:
                (
                    self._projection,
                    self._image_overlay,
                    self._accepted_fit,
                    self._facet_thresholds,
                    self._focused_facet_index,
                    self._facet_focus_index,
                    self._viewport,
                    self._layout_revision,
                ) = previous
            try:
                if plan is not None:
                    self._apply_layout_plan(
                        old_plan,
                        schedule_fit=False,
                        notify_surface=False,
                    )
                else:
                    self._render_current(
                        RenderEffect.BASE_GEOMETRY,
                        schedule_fit=False,
                    )
            except BaseException:
                pass
            raise
        return _ProjectionPresentation(
            projection,
            *previous,
            old_plan,
            callbacks,
        )

    def _abort_projection_presentation(
        self,
        presentation: _ProjectionPresentation,
    ) -> None:
        """Restore a drawn projection that never reached the frontend."""

        if not isinstance(presentation, _ProjectionPresentation):
            raise TypeError("presentation must be a projection presentation")
        with self._render_lock:
            with self._lock:
                if self._projection is not presentation.committed_projection:
                    raise RuntimeError(
                        "projection presentation is no longer current"
                    )
                current_plan = self.surface_plan
                self._projection = presentation.previous_projection
                self._image_overlay = presentation.previous_image_overlay
                self._accepted_fit = presentation.previous_accepted_fit
                self._facet_thresholds = presentation.previous_facet_thresholds
                self._focused_facet_index = (
                    presentation.previous_focused_facet_index
                )
                self._facet_focus_index = presentation.previous_facet_focus_index
                self._viewport = presentation.previous_viewport
                self._layout_revision = presentation.previous_layout_revision
            if current_plan != presentation.previous_plan:
                self._apply_layout_plan(
                    presentation.previous_plan,
                    schedule_fit=False,
                    notify_surface=False,
                )
            else:
                self._render_current(
                    RenderEffect.BASE_GEOMETRY,
                    schedule_fit=False,
                )

    def _apply_layout_plan(
        self,
        plan: SurfacePlan,
        *,
        schedule_fit: bool = False,
        notify_surface: bool = True,
    ) -> tuple[SurfaceCallback, ...]:
        if not isinstance(notify_surface, bool):
            raise TypeError("notify_surface must be bool")
        with self._render_lock:
            self._cancel_gesture()
            assert self._renderer is not None
            renderer = self._renderer
            renderer.relayout(
                plan,
                facet_index=self._focused_facet_index,
                facet_focus_index=self._facet_focus_index,
            )
            self._update_renderer(renderer, RenderEffect.LAYOUT)
            with self._lock:
                self._assert_open()
                callbacks = tuple(self._surface_callbacks)
        if notify_surface:
            self._notify_surface_callbacks(callbacks)
        return callbacks

    def set_parameter(self, name: str, value: object) -> DisplayState:
        return self.set_parameters({name: value})

    def set_labels(
        self,
        *,
        title: str | None | object = _UNSET,
        x: str | None | object = _UNSET,
        y: str | None | object = _UNSET,
        value: str | None | object = _UNSET,
    ) -> DisplayState:
        """Update visible text artists without rebuilding data artists.

        ``None`` clears the title and resets an axis/value label to its
        data-declared automatic text.  Use an empty string to hide an axis or
        value label deliberately.
        """

        updates: dict[str, object] = {}
        for name, selected in (
            ("title", title),
            ("x_label", x),
            ("y_label", y),
            ("value_label", value),
        ):
            if selected is _UNSET:
                continue
            updates[name] = "" if name == "title" and selected is None else selected
        if not updates:
            return self.display_state
        return self.set_parameters(updates)

    def _prepare_value_unit_ranges(
        self,
        prepared: dict[str, object],
        previous: DisplayState,
        authored_names: frozenset[str],
    ) -> None:
        if (
            self._view is None
            or not isinstance(self._projection.data, DatasetSnapshot)
            or "value_display_unit" not in prepared
            or prepared["value_display_unit"] == previous.values.get("value_display_unit")
        ):
            return
        if (
            prepared.get("relim_mode") != "fixed"
            and "relim_mode" in prepared
        ):
            return
        semantic = self._projected._semantic_spec()
        if isinstance(semantic, (CurvePlot, RollingPlot)):
            range_names = ("y_min", "y_max")
        elif isinstance(semantic, ImagePlot):
            range_names = ("color_min", "color_max")
        else:
            return
        selected_unit = prepared["value_display_unit"]
        target_unit = (
            self._projection.data.schema.display_unit
            if selected_unit is None
            else resolve_unit(selected_unit, self._unit_registry)
        )
        source_unit = self._projected._value_quantity().display_unit
        for name in range_names:
            if name not in self._parameter_schema:
                continue
            if name in authored_names:
                continue
            current = prepared.get(name, previous.values.get(name))
            if current is None:
                continue
            converted = source_unit.convert_value_to((float(current),), target_unit)
            prepared[name] = float(np.asarray(converted).reshape(-1)[0])

    def _materialize_fixed_limits(
        self,
        prepared: dict[str, object],
        previous: DisplayState,
    ) -> None:
        if "relim_mode" not in self._parameter_schema:
            return
        mode = prepared.get("relim_mode", previous.values["relim_mode"])
        if mode != "fixed":
            return
        candidate = dict(previous.values)
        candidate.update(prepared)
        assert self._renderer is not None
        for low_name in self._parameter_schema.names:
            if not low_name.endswith("_min"):
                continue
            high_name = f"{low_name[:-4]}_max"
            if high_name not in self._parameter_schema:
                continue
            if candidate[low_name] is not None and candidate[high_name] is not None:
                continue
            if low_name == "color_min":
                low, high = sorted(self._renderer.resolved_color_limits())
            else:
                low, high = sorted(map(float, self._renderer.primary_axes.get_ylim()))
            if candidate[low_name] is None:
                prepared[low_name] = low
            if candidate[high_name] is None:
                prepared[high_name] = high

    def _validate_projection_unit_updates(
        self,
        prepared: Mapping[str, object],
    ) -> None:
        """Resolve unit changes before committing display state."""

        if self._view is None:
            return
        for name, quantity in self._unit_parameter_sources().items():
            if name not in prepared or prepared[name] is None:
                continue
            target = resolve_unit(prepared[name], self._unit_registry)
            if not quantity.canonical_unit.compatible_with(target):
                raise ValueError(
                    f"{name} {target.symbol!r} is incompatible with "
                    f"{quantity.canonical_unit.symbol!r}"
                )

    def _invalidate_fit_context(self) -> int:
        with self._lock:
            self._fit_context_generation += 1
            self._fit_cancel.set()
            if self._live_fit_request is not None and not self._closed:
                self._live_fit_pending = True
            else:
                self._clear_fit_presentation()
            return self._fit_context_generation

    def _clear_fit_presentation(self) -> bool:
        """Clear the one accepted result/selection/overlay state group."""

        changed = (
            self._accepted_fit is not None or self._accepted_facet_fit is not None
        )
        self._accepted_fit = None
        self._accepted_facet_fit = None
        return changed

    def set_parameters(self, values: Mapping[str, object]) -> DisplayState:
        surface_callbacks: tuple[SurfaceCallback, ...] = ()
        with self._render_lock:
            with self._lock:
                self._assert_open()
                prepared = self._parameter_schema.prepare_updates(values)
                authored_names = frozenset(
                    name for name, value in prepared.items() if value is not None
                )
                previous = self.display_state
                self._validate_projection_unit_updates(prepared)
                self._materialize_fixed_limits(prepared, previous)
                self._prepare_value_unit_ranges(
                    prepared,
                    previous,
                    authored_names,
                )
                candidate = self._parameter_schema._transition_prepared(
                    previous.values,
                    prepared,
                )
                accepted_changes = frozenset(
                    name
                    for name in self._parameter_schema.names
                    if previous.values[name] != candidate[name]
                )
                if not accepted_changes:
                    return previous
                accepted_effects = self._parameter_schema.effects_for(
                    accepted_changes
                )
                unit_affecting = bool(
                    accepted_effects & RenderEffect.VIEW_PROJECTION
                )
                canonical_viewport = (
                    self._projected._viewport_in_canonical()
                    if unit_affecting
                    and self._viewport is not None
                    and self._view is not None
                    else None
                )
                if accepted_effects & (
                    RenderEffect.INTERACTION_REPROJECT | RenderEffect.LAYOUT
                ):
                    self._cancel_gesture()
                old_plan = self.surface_plan
                previous_projection = self._projection._with_context(
                    self._projection_context()
                )
                previous_values = (
                    self._viewport,
                    self._accepted_fit,
                    self._accepted_facet_fit,
                    self._fit_context_generation,
                    self._live_fit_pending,
                    self._layout_revision,
                )
                state = self._display_store._commit_prepared(previous, candidate)
                fit_cancel: Event | None = None
                layout_attempted = False
                try:
                    changed = state.changed_names
                    if (
                        isinstance(self._spec, PulseTimelinePlot)
                        and "x_display_unit" in changed
                        and self._viewport is not None
                    ):
                        assert isinstance(self._projection.data, PulseTimelineData)
                        old_factor, _old_unit = pulse_time_scale(
                            self._projection.data,
                            previous.values.get("x_display_unit"),
                        )
                        new_factor, _new_unit = pulse_time_scale(
                            self._projection.data,
                            state.values.get("x_display_unit"),
                        )
                        source_x = NumericRange(
                            self._viewport.x.low / old_factor,
                            self._viewport.x.high / old_factor,
                        )
                        self._viewport = RectangleRange(
                            NumericRange(
                                source_x.low * new_factor,
                                source_x.high * new_factor,
                            ),
                            self._viewport.y,
                        )
                    effects = state.effects
                    unit_projection_changed = bool(
                        effects & RenderEffect.VIEW_PROJECTION
                    )
                    payload_projection_changed = bool(
                        effects & RenderEffect.PAYLOAD_PROJECTION
                    )
                    fit_selection_changed = bool(
                        effects & RenderEffect.FIT_SELECTION
                    )
                    if fit_selection_changed:
                        self._fit_context_generation += 1
                        fit_cancel = self._fit_cancel
                        if self._live_fit_request is not None and not self._closed:
                            self._live_fit_pending = True
                        else:
                            self._clear_fit_presentation()
                    if unit_projection_changed:
                        self._rebuild_projection()
                        if canonical_viewport is not None:
                            self._viewport = self._viewport_from_canonical(
                                canonical_viewport
                            )
                    elif payload_projection_changed:
                        self._rebuild_projection(payload_only=True)
                    if fit_selection_changed:
                        self._clear_fit_presentation()
                    elif unit_affecting and self._accepted_fit is not None:
                        accepted = self._accepted_fit
                        self._accepted_fit = replace(
                            accepted,
                            overlay=self._projected._make_fit_overlay(
                                accepted.result,
                                accepted.selection,
                            ),
                        )
                    elif unit_affecting and self._accepted_facet_fit is not None:
                        accepted_batch = self._accepted_facet_fit
                        self._accepted_facet_fit = replace(
                            accepted_batch,
                            overlays=tuple(
                                self._projected._make_fit_overlay(result, selection)
                                for result, selection in zip(
                                    accepted_batch.result.results,
                                    accepted_batch.selections,
                                    strict=True,
                                )
                                if result is not None
                            ),
                        )
                    plan = (
                        self._resolve_plan()
                        if effects & RenderEffect.LAYOUT
                        else None
                    )
                    if plan is not None:
                        self._layout_revision += 1
                except BaseException:
                    self._display_store._restore_prepared(state, previous)
                    self._projection = previous_projection
                    (
                        self._viewport,
                        self._accepted_fit,
                        self._accepted_facet_fit,
                        self._fit_context_generation,
                        self._live_fit_pending,
                        self._layout_revision,
                    ) = previous_values
                    raise
            try:
                if plan is not None:
                    layout_attempted = True
                    surface_callbacks = self._apply_layout_plan(
                        plan,
                        schedule_fit=False,
                        notify_surface=False,
                    )
                else:
                    self._render_current(
                        state.effects,
                        schedule_fit=False,
                    )
            except BaseException:
                with self._lock:
                    self._display_store._restore_prepared(state, previous)
                    self._projection = previous_projection
                    (
                        self._viewport,
                        self._accepted_fit,
                        self._accepted_facet_fit,
                        self._fit_context_generation,
                        self._live_fit_pending,
                        self._layout_revision,
                    ) = previous_values
                try:
                    if layout_attempted or self.surface_plan != old_plan:
                        self._apply_layout_plan(
                            old_plan,
                            schedule_fit=False,
                            notify_surface=False,
                        )
                    else:
                        self._render_current(
                            RenderEffect.LAYOUT,
                            schedule_fit=False,
                        )
                except BaseException:
                    pass
                raise
            if fit_cancel is not None:
                fit_cancel.set()
        self._notify_surface_callbacks(surface_callbacks)
        self._notify_display(state)
        return state

    def set_size(self, preset: str) -> SurfacePlan:
        selected = self._defaults.layout.validate_preset(preset)
        callbacks: tuple[SurfaceCallback, ...] = ()
        with self._render_lock:
            with self._lock:
                self._assert_open()
                if selected == self._size:
                    return self.surface_plan
                self._cancel_gesture()
                previous_size = self._size
                previous_revision = self._layout_revision
                previous_plan = self.surface_plan
                try:
                    self._size = selected
                    plan = self._resolve_plan()
                    self._layout_revision += 1
                except BaseException:
                    self._size = previous_size
                    self._layout_revision = previous_revision
                    raise
            try:
                callbacks = self._apply_layout_plan(
                    plan,
                    schedule_fit=False,
                    notify_surface=False,
                )
            except BaseException:
                with self._lock:
                    self._size = previous_size
                    self._layout_revision = previous_revision
                try:
                    self._apply_layout_plan(
                        previous_plan,
                        schedule_fit=False,
                        notify_surface=False,
                    )
                except BaseException:
                    pass
                raise
        self._notify_surface_callbacks(callbacks)
        return plan

    def focus_facet(self, index: int) -> None:
        """Open one FacetGrid cell as the full interactive plot surface."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            target_changed = self._select_facet(index)
            assert self._renderer is not None
            presentation_changed = self._facet_focus_index != index
            if presentation_changed:
                self._cancel_gesture()
                self._facet_focus_index = index
            if target_changed or presentation_changed:
                self._render_current(
                    RenderEffect.BASE_GEOMETRY
                    | RenderEffect.AXIS_TRANSFORM
                    | RenderEffect.CHROME
                )

    def show_facet_overview(self) -> None:
        """Return a focused FacetGrid cell to its non-interactive overview."""

        if not isinstance(self._spec, FacetGridPlot):
            raise TypeError("facet overview is available only for FacetGridPlot")
        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            if self._facet_focus_index is None:
                return
            self._cancel_gesture()
            cleared_viewport = self._viewport is not None
            self._viewport = None
            if cleared_viewport:
                self._invalidate_fit_context()
                self._clear_fit_presentation()
            self._facet_focus_index = None
            self._render_current(
                RenderEffect.BASE_GEOMETRY
                | RenderEffect.AXIS_TRANSFORM
                | RenderEffect.CHROME
            )

    def _select_facet(self, index: int) -> bool:
        """Route cell-local state without changing overview/focus presentation."""

        if not isinstance(self._spec, FacetGridPlot):
            raise TypeError("facet selection is available only for FacetGridPlot")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("facet index must be an integer")
        cells = tuple(getattr(self._payload, "cells", ()))
        if index < 0 or index >= len(cells):
            raise IndexError("facet index is outside the current grid")
        changed = index != self._focused_facet_index
        if changed:
            self._cancel_gesture()
            preserved_batch = (
                self._accepted_facet_fit if self._viewport is None else None
            )
            self._focused_facet_index = index
            self._viewport = None
            self._invalidate_fit_context()
            self._clear_fit_presentation()
            if preserved_batch is not None:
                self._accepted_facet_fit = replace(
                    preserved_batch,
                    context_generation=self._fit_context_generation,
                )
        return changed

    def _clamp_facet_state(self, cell_count: int) -> None:
        """Keep selected and open cells valid after a payload topology change."""

        if cell_count <= 0:
            selected = None
            opened = None
        else:
            selected = (
                cell_count - 1
                if self._focused_facet_index is None
                else min(self._focused_facet_index, cell_count - 1)
            )
            opened = selected if self._facet_focus_index is not None else None
        if (
            selected != self._focused_facet_index
            or opened != self._facet_focus_index
        ):
            self._focused_facet_index = selected
            self._facet_focus_index = opened
            self._viewport = None

    def set_device_pixel_ratio(self, ratio: float) -> SurfacePlan:
        return self._set_device_pixel_ratio(ratio, preserve_native_canvas=False)

    def _set_device_pixel_ratio(
        self,
        ratio: float,
        *,
        preserve_native_canvas: bool,
    ) -> SurfacePlan:
        selected = _validated_device_pixel_ratio(ratio)
        callbacks: tuple[SurfaceCallback, ...] = ()
        with self._render_lock:
            with self._lock:
                self._assert_open()
                if selected == self._device_pixel_ratio:
                    return self.surface_plan
                self._cancel_gesture()
                previous_ratio = self._device_pixel_ratio
                previous_revision = self._layout_revision
                previous_plan = self.surface_plan
                try:
                    self._device_pixel_ratio = selected
                    plan = self._resolve_plan()
                    self._layout_revision += 1
                    if preserve_native_canvas:
                        assert self._renderer is not None
                        self._renderer.plan = plan
                        figure = self._renderer.figure
                        figure._original_dpi = plan.logical_dpi
                        figure._set_dpi(plan.dpi, forward=False)
                except BaseException:
                    self._device_pixel_ratio = previous_ratio
                    self._layout_revision = previous_revision
                    raise
            try:
                if preserve_native_canvas:
                    self._render_current(
                        RenderEffect.LAYOUT,
                        schedule_fit=False,
                    )
                else:
                    callbacks = self._apply_layout_plan(
                        plan,
                        schedule_fit=False,
                        notify_surface=False,
                    )
            except BaseException:
                with self._lock:
                    self._device_pixel_ratio = previous_ratio
                    self._layout_revision = previous_revision
                    assert self._renderer is not None
                    self._renderer.plan = previous_plan
                    figure = self._renderer.figure
                    figure._original_dpi = previous_plan.logical_dpi
                    figure._set_dpi(previous_plan.dpi, forward=False)
                try:
                    if preserve_native_canvas:
                        self._render_current(
                            RenderEffect.LAYOUT,
                            schedule_fit=False,
                        )
                    else:
                        self._apply_layout_plan(
                            previous_plan,
                            schedule_fit=False,
                            notify_surface=False,
                        )
                except BaseException:
                    pass
                raise
        self._notify_surface_callbacks(callbacks)
        return plan

    def adopt_native_device_pixel_ratio(self, ratio: float) -> SurfacePlan:
        """Adopt DPR already applied by an attached native canvas in place.

        A browser/Qt adapter may report a new DPR while its view is still
        materialising.  Replacing that widget at this point invalidates its
        child controls.  Logical geometry and axes topology do not change with
        DPR, so the existing renderer can safely adopt the new physical plan
        without replacing the Figure or canvas.
        """

        return self._set_device_pixel_ratio(ratio, preserve_native_canvas=True)

    def set_axis_unit(self, axis: AxisRef, unit: str | None) -> DisplayState:
        if not isinstance(axis, AxisRef):
            raise TypeError("axis must be AxisRef")
        target_name: str | None = None
        semantic = self._spec.cell if isinstance(self._spec, FacetGridPlot) else self._spec
        if getattr(semantic, "x", None) == axis:
            target_name = "x_display_unit"
        elif getattr(semantic, "y", None) == axis:
            target_name = "y_display_unit"
        elif isinstance(self._spec, FacetGridPlot) and self._spec.facet == axis:
            target_name = "facet_display_unit"
        if target_name is None:
            raise ValueError("axis is not a displayed x, y or facet axis in this plot")
        if target_name not in self._parameter_schema:
            raise TypeError(
                f"this plot kind does not expose {target_name!r}"
            )
        return self.set_parameter(target_name, unit)

    def set_value_unit(self, unit: str | None) -> DisplayState:
        return self.set_parameter("value_display_unit", unit)

    def set_time_unit(self, unit: str | None) -> DisplayState:
        """Set a PulseTimeline display unit, or ``None`` for automatic scaling."""

        if not isinstance(self._spec, PulseTimelinePlot):
            raise TypeError("set_time_unit is available only for PulseTimelinePlot")
        return self.set_parameter("x_display_unit", unit)

    def set_color_limits(
        self,
        low: float,
        high: float,
        *,
        fixed: bool = True,
    ) -> DisplayState:
        """Atomically edit an Image or Facet-image color range."""

        if "color_min" not in self._parameter_schema:
            raise TypeError("this plot kind has no color limits")
        values: dict[str, object] = {"color_min": low, "color_max": high}
        if fixed:
            values["relim_mode"] = "fixed"
        return self.set_parameters(values)

    def resolved_color_limits(self, *, display: bool = True) -> NumericRange:
        """Return the color range painted by the current image front."""

        if not isinstance(display, bool):
            raise TypeError("display must be bool")
        if "color_min" not in self._parameter_schema:
            raise TypeError("this plot kind has no color limits")
        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            value = NumericRange(*self._renderer.resolved_color_limits())
            if display or self._view is None:
                return value
            return self._projected._display_range_to_canonical(
                value,
                self._projected._value_quantity(),
            )

    def set_relim_mode(self, mode: str) -> DisplayState:
        """Select tight, normal, or fixed automatic axis scaling."""

        if "relim_mode" not in self._parameter_schema:
            raise TypeError("this plot kind has no relim mode")
        return self.set_parameter("relim_mode", mode)

    def set_y_limits(
        self,
        low: float,
        high: float,
        *,
        fixed: bool = True,
    ) -> DisplayState:
        """Atomically edit a curve value or histogram count y range."""

        if "y_min" not in self._parameter_schema:
            raise TypeError("this plot kind has no configurable y limits")
        values: dict[str, object] = {"y_min": low, "y_max": high}
        if fixed:
            values["relim_mode"] = "fixed"
        return self.set_parameters(values)

    def reset_y_limits(self, *, mode: str = "normal") -> DisplayState:
        if "y_min" not in self._parameter_schema:
            raise TypeError("this plot kind has no configurable y limits")
        return self.set_parameters(
            {"relim_mode": mode, "y_min": None, "y_max": None}
        )

    def reset_color_limits(self, *, mode: str = "tight") -> DisplayState:
        if "color_min" not in self._parameter_schema:
            raise TypeError("this plot kind has no color limits")
        return self.set_parameters(
            {"relim_mode": mode, "color_min": None, "color_max": None}
        )

    def set_x_limits(self, low: float, high: float) -> RectangleRange:
        """Set the visible x range in the current display unit."""

        selected_x = NumericRange(float(low), float(high))
        with self._render_lock:
            with self._lock:
                self._assert_open()
            current = self._current_display_limits()
        return self.set_viewport(selected_x, current.y)

    def set_view_limits(
        self,
        *,
        x: tuple[float, float] | NumericRange | None = None,
        y: tuple[float, float] | NumericRange | None = None,
    ) -> RectangleRange:
        """Set either or both visible ranges in current display units."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            current = self._current_display_limits()

        def selected_range(
            value: tuple[float, float] | NumericRange | None,
            fallback: NumericRange,
            name: str,
        ) -> NumericRange:
            if value is None:
                return fallback
            if isinstance(value, NumericRange):
                return value
            try:
                low_value, high_value = value
            except (TypeError, ValueError) as error:
                raise TypeError(f"{name} limits must contain two values") from error
            return NumericRange(float(low_value), float(high_value))

        return self.set_viewport(
            selected_range(x, current.x, "x"),
            selected_range(y, current.y, "y"),
        )

    def update_data(
        self,
        data: PlotInput,
        *,
        revision: int | None = None,
    ) -> None:
        """Present new data, optionally preserving a producer-owned revision.

        DatasetSnapshot keeps its intrinsic revision. PulseTimeline callers may
        provide the revision from a live transport envelope; direct
        calls without one advance the current session revision by exactly one.
        Once automatic live fit is armed, callers must publish through
        :class:`LivePlotController`, which prepares matching data and fit before
        either becomes visible; synchronous ``update_data`` is rejected rather
        than blocking an owner/UI thread or painting a mismatched frame.
        """

        data, image_frame = self._split_image_frame(data, self._spec)
        image_overlay = _UNSET if image_frame is None else image_frame.overlay
        FitProjection._validate_input(data, self._spec)
        if revision is not None:
            if isinstance(revision, bool) or not isinstance(revision, Integral):
                raise TypeError("revision must be an integer or None")
            revision = int(revision)
            if revision < 0:
                raise ValueError("revision must be non-negative")
        if image_frame is not None and revision is not None:
            raise ValueError("ImageFrame revision is owned by its snapshot")

        with self._render_lock:
            with self._lock:
                self._assert_open()
                if isinstance(data, DatasetSnapshot):
                    assert isinstance(self._projection.data, DatasetSnapshot)
                    if revision is not None and revision != data.revision:
                        raise ValueError(
                            "DatasetSnapshot revision must equal the supplied revision"
                        )
                    self._projection.data.schema.assert_exact(data.schema)
                    if data.revision <= self._projection.data.revision:
                        raise ValueError(
                            "data revision must increase: "
                            f"{data.revision} <= {self._projection.data.revision}"
                        )
                    if image_frame is not None:
                        self._validate_image_frame_overlay(
                            self._image_overlay,
                            image_frame.overlay,
                        )
                else:
                    next_revision = (
                        self.data_revision + 1
                        if revision is None
                        else revision
                    )
                    if next_revision <= self.data_revision:
                        raise ValueError(
                            "data revision must increase: "
                            f"{next_revision} <= {self.data_revision}"
                        )
                selected_revision = (
                    data.revision
                    if isinstance(data, DatasetSnapshot)
                    else next_revision
                )
                projection = self._projection._fork_frozen(
                    data=data,
                    revision=selected_revision,
                    context=self._projection_context(),
                )
                projection._build_view_and_payload()
                accepted_overlay = (
                    self._image_overlay
                    if image_overlay is _UNSET
                    else image_overlay
                )
            presentation = self._present_projection_transaction(
                projection,
                image_overlay=accepted_overlay,
                accepted_fit=None,
            )
            self._invalidate_fit_context()
        self._notify_surface_callbacks(presentation.surface_callbacks)
        self._start_pending_live_fit()

    def update_image_frame(self, frame: ImageFrame) -> ImageFrame:
        """Present image data and its point layer in one render transaction."""

        if not isinstance(frame, ImageFrame):
            raise TypeError("frame must be ImageFrame")
        if not isinstance(self._spec, ImagePlot):
            raise TypeError("ImageFrame requires ImagePlot")
        self.update_data(frame)
        return frame

    @property
    def image_overlay(self) -> ImagePointOverlay | None:
        """Latest immutable canonical point layer for an Image plot."""

        with self._lock:
            return self._image_overlay

    @property
    def image_overlay_revision(self) -> int | None:
        with self._lock:
            overlay = self._image_overlay
            return None if overlay is None else overlay.revision

    def update_image_overlay(self, overlay: ImagePointOverlay) -> ImagePointOverlay:
        """Present one strictly newer point-layer revision without touching image data."""

        if not isinstance(overlay, ImagePointOverlay):
            raise TypeError("overlay must be ImagePointOverlay")
        if not isinstance(self._spec, ImagePlot):
            raise TypeError("image point overlays require ImagePlot")
        with self._render_lock:
            with self._lock:
                self._assert_open()
                previous = self._image_overlay
                if previous is not None and overlay.revision <= previous.revision:
                    raise RevisionError(
                        "image overlay revision must strictly increase"
                    )
                self._image_overlay = overlay
            try:
                self._render_current(RenderEffect.OVERLAY)
            except BaseException:
                self._image_overlay = previous
                try:
                    self._render_current(RenderEffect.OVERLAY)
                except BaseException:
                    pass
                raise
        return overlay

    def subscribe_surface(self, callback: SurfaceCallback) -> Callable[[], None]:
        return self._subscribe_callback(self._surface_callbacks, callback)

    def _subscribe_callback(
        self,
        callbacks: list[_CallbackT],
        callback: _CallbackT,
    ) -> Callable[[], None]:
        """Register one observer and return its idempotent release edge."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._assert_open()
            callbacks.append(callback)

        released = False

        def unsubscribe() -> None:
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                if callback in callbacks:
                    callbacks.remove(callback)

        return unsubscribe

    def subscribe_display(self, callback: DisplayCallback) -> Callable[[], None]:
        """Observe accepted display-state changes on the attached host thread."""

        return self._subscribe_callback(self._display_callbacks, callback)

    def _notify_display(self, state: DisplayState) -> None:
        with self._lock:
            callbacks = tuple(self._display_callbacks)
        self._notify_callbacks(callbacks, state)

    @staticmethod
    def _notify_surface_callbacks(
        callbacks: Sequence[SurfaceCallback],
    ) -> None:
        """Notify independent surface observers after a front is committed."""

        for callback in callbacks:
            try:
                callback()
            except BaseException:
                continue

    def _notify_callbacks(
        self,
        callbacks: Sequence[Callable[[_EventT], object]],
        event: _EventT,
    ) -> None:
        """Marshal one event to isolated application observers."""

        if not callbacks:
            return

        def invoke() -> None:
            for callback in callbacks:
                try:
                    callback(event)
                except BaseException:
                    # One application callback must not disable later observers.
                    continue

        self.owner_dispatch(invoke)

    def owner_dispatch(self, callback: Callable[[], _ResultT]) -> Future[_ResultT]:
        """Run through the current owner and always return its completion.

        This bound method is a stable gateway: callers may retain it before an
        interactive host is attached.  Host attachment/release and the
        headless direct path share one ownership gate, so a direct callback
        cannot overlap a transition to a notebook or raster owner.
        """

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._ownership_gate:
            with self._lock:
                dispatch = self._dispatch
            if dispatch is None:
                completion: Future[_ResultT] = Future()
                if not completion.set_running_or_notify_cancel():
                    return completion
                try:
                    value = callback()
                except BaseException as error:
                    completion.set_exception(error)
                else:
                    completion.set_result(value)
                return completion
            try:
                completion = dispatch(callback)
            except BaseException as error:
                failed: Future[_ResultT] = Future()
                failed.set_exception(error)
                return failed
            if not isinstance(completion, Future):
                invalid: Future[_ResultT] = Future()
                invalid.set_exception(
                    TypeError("host dispatch must return concurrent.futures.Future")
                )
                return invalid
            return completion

    def attach_host(
        self,
        owner: object,
        dispatch: HostDispatch,
        *,
        presentation_dispatch: HostPresentationDispatch | None = None,
    ) -> Callable[[], None]:
        """Attach exactly one interactive canvas host and return its release edge."""

        if owner is None:
            raise TypeError("host owner must not be None")
        if not callable(dispatch):
            raise TypeError("dispatch must be callable")
        if presentation_dispatch is not None and not callable(
            presentation_dispatch
        ):
            raise TypeError("presentation_dispatch must be callable or None")
        with self._ownership_gate:
            with self._lock:
                self._assert_open()
                if self._host_owner is not None:
                    if self._host_owner is owner:
                        raise RuntimeError("interactive host is already attached")
                    raise RuntimeError(
                        "PlotSession supports one interactive host; create another "
                        "session over the same immutable snapshot for a second view"
                    )
                self._host_owner = owner
                self._host_previous_dispatch = self._dispatch
                self._host_previous_presentation_dispatch = (
                    self._presentation_dispatch
                )
                self._dispatch = dispatch
                self._presentation_dispatch = presentation_dispatch

        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            with self._ownership_gate:
                with self._render_lock:
                    with self._lock:
                        if self._host_owner is not owner:
                            return
                    if not self._closed:
                        self._cancel_gesture()
                    with self._lock:
                        previous = self._host_previous_dispatch
                        previous_presentation = (
                            self._host_previous_presentation_dispatch
                        )
                        self._host_owner = None
                        self._host_previous_dispatch = None
                        self._host_previous_presentation_dispatch = None
                        self._dispatch = previous
                        self._presentation_dispatch = previous_presentation

        return release

    def set_x_selector(
        self,
        low: float,
        high: float,
        *,
        display: bool = True,
    ) -> SelectorState:
        value = NumericRange(low, high)
        with self._render_lock:
            with self._lock:
                self._assert_open()
            if display:
                if self._view is not None:
                    value = self._projected._display_range_to_canonical(
                        value, self._projected._x_selector_source()
                    )
                elif isinstance(self._spec, PulseTimelinePlot):
                    value = self._pulse_display_range_to_source(value)
            return self._install_selector_state(SelectorState(
                SelectorKind.X_RANGE,
                value,
                facet_index=self._focused_facet_index,
            ))

    def set_area_selector(
        self,
        x: NumericRange,
        y: NumericRange,
        *,
        display: bool = True,
    ) -> SelectorState:
        value = RectangleRange(x, y)
        with self._render_lock:
            with self._lock:
                self._assert_open()
            if display:
                value = self._area_display_to_canonical(value)
            return self._install_selector_state(SelectorState(
                SelectorKind.AREA,
                value,
                facet_index=self._focused_facet_index,
            ))

    def _install_selector_state(
        self,
        state: SelectorState,
        *,
        emit_change: bool = True,
        finished_gesture: _SelectorGesture | None = None,
    ) -> SelectorState:
        """Atomically replace the one selector owned by ``state.kind``."""

        self._require_stable_selector(state)
        with self._render_lock:
            with self._lock:
                self._assert_open()
                if (
                    finished_gesture is None
                    and isinstance(self._gesture, _SelectorGesture)
                ):
                    self._cancel_gesture()
                previous, stored = (
                    self._selector_controller.install(state)
                    if finished_gesture is None
                    else self._selector_controller._commit_finished(state)
                )
            try:
                self._render_current(
                    RenderEffect.OVERLAY,
                    schedule_fit=False,
                )
            except BaseException:
                with self._lock:
                    self._selector_controller._rollback_install(stored, previous)
                try:
                    self._render_current(
                        RenderEffect.OVERLAY,
                        schedule_fit=False,
                    )
                except BaseException:
                    pass
                raise
        if emit_change:
            self._emit_selection(
                SelectionChange.ADDED if previous is None else SelectionChange.UPDATED,
                stored,
            )
        return stored

    def set_threshold_selector(
        self, value: float, *, display: bool = True
    ) -> SelectorState:
        with self._render_lock:
            with self._lock:
                self._assert_open()
            canonical = (
                self._projected._display_scalar_to_canonical(
                    value,
                    self._projected._value_quantity(),
                )
                if display and self._view is not None
                else value
            )
            return self._install_selector_state(SelectorState(
                SelectorKind.THRESHOLD,
                canonical,
                facet_index=self._focused_facet_index,
            ))

    def set_facet_thresholds(
        self,
        thresholds: Sequence[float | None],
        *,
        display: bool = True,
    ) -> tuple[float | None, ...]:
        """Set authoritative Histogram annotations in current facet-cell order.

        These values annotate an exact static result (for example a calibrated
        per-site threshold).  They are independent of the display-only
        threshold suggested by a Histogram fit and are cleared when either the
        source revision or PlotSpec changes.
        """

        if not isinstance(display, bool):
            raise TypeError("display must be bool")
        with self._render_lock:
            with self._lock:
                self._assert_open()
                if not isinstance(self._spec, FacetGridPlot) or not isinstance(
                    self._spec.cell,
                    HistogramPlot,
                ):
                    raise TypeError(
                        "facet thresholds require FacetGridPlot[HistogramPlot]"
                    )
                cells = tuple(getattr(self._payload, "cells", ()))
                prepared = tuple(thresholds)
                if len(prepared) != len(cells):
                    raise ValueError(
                        "facet thresholds must match the current FacetData cell order"
                    )
                canonical: list[float | None] = []
                for value in prepared:
                    if value is None:
                        canonical.append(None)
                        continue
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        raise ValueError("facet thresholds must be finite or None")
                    canonical.append(
                        self._projected._display_scalar_to_canonical(
                            numeric,
                            self._projected._value_quantity(),
                        )
                        if display
                        else numeric
                    )
                selected = tuple(canonical)
                previous = self._facet_thresholds
                self._facet_thresholds = selected
            try:
                self._render_current(
                    RenderEffect.OVERLAY,
                    schedule_fit=False,
                )
            except BaseException:
                with self._lock:
                    self._facet_thresholds = previous
                try:
                    self._render_current(
                        RenderEffect.OVERLAY,
                        schedule_fit=False,
                    )
                except BaseException:
                    pass
                raise
        return selected

    def set_crosshair_selector(
        self, x: float, y: float, *, display: bool = True
    ) -> SelectorState:
        with self._render_lock:
            with self._lock:
                self._assert_open()
            point = CrosshairPoint(x, y)
            if display:
                if self._view is not None:
                    point = CrosshairPoint(
                        self._display_x_scalar_to_canonical(x),
                        y
                        if self._projected._is_histogram_plot()
                        else self._projected._display_scalar_to_canonical(
                            y, self._projected._y_ref_or_value()
                        ),
                    )
                elif isinstance(self._spec, PulseTimelinePlot):
                    point = CrosshairPoint(self._pulse_display_x_to_source(x), y)
            return self._install_selector_state(SelectorState(
                SelectorKind.CROSSHAIR,
                point,
                facet_index=self._focused_facet_index,
            ))

    def selector_state(
        self,
        kind: SelectorKind,
        *,
        display: bool = False,
    ) -> SelectorState:
        with self._render_lock:
            with self._lock:
                self._assert_open()
            state = next(
                (
                    item
                    for item in self._resolved_selector_snapshot().states
                    if item.kind is kind
                ),
                None,
            )
            if state is None:
                raise KeyError(kind)
            if not display:
                return state
            if self._view is not None:
                return self._projected._display_selector_state(state)
            return self._special_display_selector_state(state)

    def set_selector_value(
        self,
        kind: SelectorKind,
        value: SelectorValue,
        *,
        display: bool = True,
    ) -> SelectorState:
        """Update a selector without exposing controller or canonical-unit details."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
            current = self._selector_controller.state(kind)
            canonical = self._canonical_selector_value(current, value, display=display)
            candidate = replace(current, value=canonical)
            self._require_stable_selector(candidate)
            return self._install_selector_state(candidate)

    def _canonical_selector_value(
        self,
        current: SelectorState,
        value: SelectorValue,
        *,
        display: bool,
    ) -> SelectorValue:
        canonical = value
        if display and self._view is not None:
            if current.kind is SelectorKind.X_RANGE:
                if not isinstance(value, NumericRange):
                    raise TypeError("x-range selector requires NumericRange")
                canonical = self._projected._display_range_to_canonical(
                    value, self._projected._x_selector_source()
                )
            elif current.kind is SelectorKind.AREA:
                if not isinstance(value, RectangleRange):
                    raise TypeError("area selector requires RectangleRange")
                canonical = self._area_display_to_canonical(value)
            elif current.kind is SelectorKind.CROSSHAIR:
                if not isinstance(value, CrosshairPoint):
                    raise TypeError("crosshair selector requires CrosshairPoint")
                canonical = CrosshairPoint(
                    self._display_x_scalar_to_canonical(value.x),
                    value.y
                    if self._projected._is_histogram_plot()
                    else self._projected._display_scalar_to_canonical(
                        value.y, self._projected._y_ref_or_value()
                    ),
                )
            elif current.kind is SelectorKind.THRESHOLD:
                canonical = self._projected._display_scalar_to_canonical(
                    float(value), self._projected._value_quantity()
                )
        elif display and isinstance(self._spec, PulseTimelinePlot):
            if current.kind is SelectorKind.X_RANGE:
                if not isinstance(value, NumericRange):
                    raise TypeError("x-range selector requires NumericRange")
                canonical = self._pulse_display_range_to_source(value)
            elif current.kind is SelectorKind.AREA:
                if not isinstance(value, RectangleRange):
                    raise TypeError("area selector requires RectangleRange")
                canonical = self._area_display_to_canonical(value)
            elif current.kind is SelectorKind.CROSSHAIR:
                if not isinstance(value, CrosshairPoint):
                    raise TypeError("crosshair selector requires CrosshairPoint")
                canonical = CrosshairPoint(
                    self._pulse_display_x_to_source(value.x), value.y
                )
        return canonical

    def commit_selector(
        self,
        kind: SelectorKind,
        value: SelectorValue,
        *,
        display: bool = True,
    ) -> SelectorState:
        """Commit one frontend gesture through the shared selector lifecycle."""

        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        with self._render_lock:
            with self._lock:
                self._assert_open()
            current = self._projected._selector_state_or_none(kind)
            if current is not None:
                canonical = self._canonical_selector_value(
                    current,
                    value,
                    display=display,
                )
                candidate = replace(
                    current,
                    value=canonical,
                    facet_index=self._focused_facet_index,
                )
                self._require_stable_selector(candidate)
                if candidate == current:
                    self._render_current(RenderEffect.OVERLAY)
                    self._emit_selection(SelectionChange.COMMITTED, current)
                    return current
                state = self._install_selector_state(
                    candidate,
                    emit_change=False,
                )
            else:
                provisional = SelectorState(
                    kind,
                    value,
                    facet_index=self._focused_facet_index,
                )
                canonical = self._canonical_selector_value(
                    provisional,
                    value,
                    display=display,
                )
                prepared = replace(provisional, value=canonical)
                state = self._install_selector_state(
                    prepared,
                    emit_change=False,
                )
            self._emit_selection(SelectionChange.COMMITTED, state)
        return state

    def remove_selector(self, kind: SelectorKind) -> SelectorState:
        """Remove a selector and notify external subscribers."""

        cancelled_fit: Future[FitResult] | None = None
        with self._render_lock:
            with self._lock:
                self._assert_open()
                state = self._selector_controller.state(kind)
                gesture = self._gesture
                if (
                    isinstance(gesture, _SelectorGesture)
                    and gesture.kind is state.kind
                ):
                    self._cancel_gesture()
                previous_fit = (
                    self._fit_context_generation,
                    self._fit_request_generation,
                    self._live_fit_completion,
                    self._live_fit_request,
                    self._live_fit_future,
                    self._live_fit_pending,
                    self._accepted_fit,
                )
                fit_cancel = self._fit_cancel
                request = self._live_fit_request
                bound_request = bool(
                    request is not None and request.selector_kind is state.kind
                )
                self._selector_controller.remove(kind)
                if bound_request:
                    self._fit_context_generation += 1
                    self._fit_request_generation += 1
                    cancelled_fit = self._live_fit_completion
                    self._live_fit_completion = None
                    self._live_fit_request = None
                    self._live_fit_future = None
                    self._live_fit_pending = False
            try:
                if self._renderer is not None:
                    self._render_current(
                        RenderEffect.OVERLAY,
                        schedule_fit=False,
                    )
            except BaseException:
                with self._lock:
                    self._selector_controller._restore_removed(state)
                    (
                        self._fit_context_generation,
                        self._fit_request_generation,
                        self._live_fit_completion,
                        self._live_fit_request,
                        self._live_fit_future,
                        self._live_fit_pending,
                        self._accepted_fit,
                    ) = previous_fit
                try:
                    if self._renderer is not None:
                        self._render_current(
                            RenderEffect.OVERLAY,
                            schedule_fit=False,
                        )
                except BaseException:
                    pass
                raise
            if bound_request:
                fit_cancel.set()
        self._emit_selection(SelectionChange.REMOVED, state)
        if cancelled_fit is not None and not cancelled_fit.done():
            cancelled_fit.set_exception(
                FitCancelled(f"fit selector removed: {state.kind.value}")
            )
        return state

    def subscribe_selection(
        self,
        callback: SelectionCallback,
        *,
        selector_kind: SelectorKind | None = None,
    ) -> Callable[[], None]:
        """Observe selector changes; COMMITTED carries its exact frozen data."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        if selector_kind is not None and not isinstance(selector_kind, SelectorKind):
            raise TypeError("selector_kind must be SelectorKind or None")
        subscription = _SelectionSubscription(callback, selector_kind)
        with self._lock:
            self._assert_open()
            self._selection_subscriptions.append(subscription)

        released = False

        def unsubscribe() -> None:
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                if subscription in self._selection_subscriptions:
                    self._selection_subscriptions.remove(subscription)

        return unsubscribe

    def _emit_selection(self, change: SelectionChange, state: SelectorState) -> None:
        with self._render_lock:
            with self._lock:
                subscriptions = tuple(
                    item
                    for item in self._selection_subscriptions
                    if item.selector_kind in (None, state.kind)
                )
                if not subscriptions:
                    return
                display_state = (
                    self._special_display_selector_state(state)
                    if self._view is None
                    else self._projected._display_selector_state(state)
                )
                data = (
                    self._selector_data_for_state(state)
                    if change is SelectionChange.COMMITTED
                    else None
                )
                event = SelectionEvent(
                    change,
                    state,
                    display_state,
                    self.data_revision,
                    data,
                )

        self._notify_callbacks(
            tuple(item.callback for item in subscriptions),
            event,
        )

    def selector_data(self, kind: SelectorKind) -> SelectorData:
        """Slice the current immutable snapshot only when explicitly called."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
                if isinstance(self._projection.data, PulseTimelineData):
                    return self._pulse_timeline_selector_data(kind)
                assert self._view is not None
                state = next(
                    (
                        item
                        for item in self._resolved_selector_snapshot().states
                        if item.kind is kind
                    ),
                    None,
                )
                if state is None:
                    raise KeyError(kind)
                return self._selector_data_for_state(state)

    def _selector_data_for_state(self, state: SelectorState) -> SelectorData:
        """Freeze selector output while the exact source front is locked."""

        if isinstance(self._projection.data, PulseTimelineData):
            return self._pulse_timeline_selector_data(state.kind)
        if isinstance(self._spec, RollingPlot):
            return self._rolling_selector_data(state)
        assert self._view is not None
        axes = self._selector_axes(state)
        transform = None if axes is None else axes.transData.transform
        mask = self._projected._selector_mask(
            state,
            point_transform=transform,
        )
        flat = np.flatnonzero(mask.reshape(-1))
        source = self._projection.data.source
        selected_value = self._selected_plot_value(state)
        return SelectionData(
            selector=state,
            selection=self._physical_selection(state, flat_indices=flat),
            selected_value=selected_value,
            source_revisions=(self.data_revision,),
            data_revision=self.data_revision,
            facet_index=state.facet_index,
            _source=source,
            _accepted_mask=mask,
        )

    def _selected_plot_value(self, state: SelectorState) -> Value | None:
        """Return the painted Cross value, never an arbitrary raw repeat."""

        if state.kind is not SelectorKind.CROSSHAIR:
            return None
        payload = self._projected._focused_payload(state.facet_index)
        point = state.value
        assert isinstance(point, CrosshairPoint)
        scalar: Any
        if hasattr(payload, "z") and hasattr(payload, "x") and hasattr(payload, "y"):
            x = np.asarray(payload.x.canonical, dtype=float).reshape(-1)
            y = np.asarray(payload.y.canonical, dtype=float).reshape(-1)
            column = int(np.argmin(np.abs(x - point.x)))
            row = int(np.argmin(np.abs(y - point.y)))
            scalar = np.asarray(payload.z.canonical)[row, column]
        elif hasattr(payload, "series") and payload.series:
            series = payload.series[0]
            x = np.asarray(series.x.canonical, dtype=float).reshape(-1)
            y = np.asarray(series.y.canonical).reshape(-1)
            index = int(np.argmin(np.abs(x - point.x)))
            scalar = y[index]
        elif hasattr(payload, "centers") and hasattr(payload, "counts"):
            x = np.asarray(payload.centers.canonical, dtype=float).reshape(-1)
            index = int(np.argmin(np.abs(x - point.x)))
            scalar = np.asarray(payload.counts).reshape(-1)[index]
        else:
            return None
        source = self._projection.data.source
        dtype = np.asarray(scalar).dtype
        return Value(
            np.asarray((scalar,), dtype=dtype),
            VALID,
            ValueSchema.scalar(dtype, source.block.schema.cell_schema.value_unit),
        )

    def _rolling_selector_data(self, state: SelectorState) -> SelectionData:
        payload = self._projection.payload
        series = tuple(getattr(payload, "series", ()))
        revisions = tuple(getattr(payload, "source_revisions", ()))
        if not series:
            raise ValueError("rolling selector requires at least one painted series")
        lengths = tuple(np.asarray(item.y.canonical).size for item in series)
        if any(length != len(revisions) for length in lengths):
            raise RuntimeError("rolling series and exact revision history disagree")
        source = self._projection.data.source
        selected_value = None
        rolling_values: np.ndarray | None = None
        rolling_valid: np.ndarray | None = None
        selected_revisions: tuple[int, ...]
        if state.kind is SelectorKind.CROSSHAIR:
            assert isinstance(state.value, CrosshairPoint)
            candidates: list[tuple[float, int, int]] = []
            all_y = np.concatenate(tuple(
                np.asarray(item.y.canonical, dtype=float).reshape(-1)
                for item in series
            ))
            y_span = max(float(np.ptp(all_y)), 1.0)
            for series_index, item in enumerate(series):
                x = np.asarray(item.x.canonical, dtype=float).reshape(-1)
                y = np.asarray(item.y.canonical, dtype=float).reshape(-1)
                valid = np.asarray(item.valid, dtype=bool).reshape(-1)
                x_span = max(float(np.ptp(x)), 1.0)
                distance = (
                    ((x - state.value.x) / x_span) ** 2
                    + ((y - state.value.y) / y_span) ** 2
                )
                distance[~valid | ~np.isfinite(distance)] = np.inf
                index = int(np.argmin(distance))
                if np.isfinite(distance[index]):
                    candidates.append((float(distance[index]), series_index, index))
            if not candidates:
                raise ValueError("rolling Cross has no valid painted sample")
            _distance, series_index, index = min(candidates)
            value = np.asarray(series[series_index].y.canonical).reshape(-1)[index]
            dtype = np.asarray(value).dtype
            selected_value = Value(
                np.asarray((value,), dtype=dtype),
                VALID,
                ValueSchema.scalar(
                    dtype,
                    source.block.schema.cell_schema.value_unit,
                ),
            )
            selected_revisions = (int(revisions[index]),)
        elif state.kind in {SelectorKind.X_RANGE, SelectorKind.AREA}:
            rows: list[tuple[int, int, object, bool]] = []
            for sample_index, revision in enumerate(revisions):
                for series_index, item in enumerate(series):
                    x = np.asarray(item.x.canonical, dtype=float).reshape(-1)
                    y = np.asarray(item.y.canonical).reshape(-1)
                    keep = False
                    if state.kind is SelectorKind.X_RANGE:
                        assert isinstance(state.value, NumericRange)
                        keep = state.value.low <= x[sample_index] <= state.value.high
                    else:
                        assert isinstance(state.value, RectangleRange)
                        numeric_y = float(y[sample_index])
                        keep = (
                            state.value.x.low <= x[sample_index] <= state.value.x.high
                            and state.value.y.low <= numeric_y <= state.value.y.high
                        )
                    if keep:
                        rows.append((
                            int(revision),
                            series_index,
                            y[sample_index],
                            bool(np.asarray(item.valid, dtype=bool)[sample_index]),
                        ))
            if not rows:
                raise ValueError("rolling selector contains no painted samples")
            selected_revisions = tuple(row[0] for row in rows)
            rolling_values = np.asarray(tuple(row[2] for row in rows))
            rolling_valid = np.asarray(tuple(row[3] for row in rows), dtype=np.bool_)
        else:
            raise TypeError(f"{state.kind.value} has no rolling data publication")
        return SelectionData(
            selector=state,
            selection=None,
            selected_value=selected_value,
            source_revisions=selected_revisions,
            data_revision=self.data_revision,
            facet_index=state.facet_index,
            _source=source,
            _rolling_values=rolling_values,
            _rolling_valid=rolling_valid,
            _rolling_unit=(
                None if rolling_values is None else series[0].y.canonical_unit.symbol
            ),
        )

    def _physical_selection(
        self,
        state: SelectorState,
        *,
        flat_indices: np.ndarray,
    ) -> Selection | None:
        """Translate visible tensor geometry once; never expose renderer masks."""

        data = self._projection.data
        if not isinstance(data, DatasetSnapshot):
            return None
        if state.kind not in {
            SelectorKind.X_RANGE,
            SelectorKind.AREA,
            SelectorKind.CROSSHAIR,
        }:
            return None
        schema = data.source.block.schema
        terms: list[IndexSelection | IndexRangeSelection] = []
        for name, ref in self._index_parameter_refs.items():
            if ref.domain is AxisDomain.REPEAT:
                axis_id = schema.repeat_axis.axis_id
            elif ref.domain is AxisDomain.DATA and ref.axis_id is not None:
                axis_id = AxisId(ref.axis_id)
            else:
                continue
            terms.append(IndexSelection(axis_id, int(self.display_state.values[name])))

        ranges: list[tuple[AxisRef, NumericRange]] = []
        x_source = self._projected._x_selector_source()
        if state.kind is SelectorKind.X_RANGE and isinstance(x_source, AxisRef):
            assert isinstance(state.value, NumericRange)
            ranges.append((x_source, state.value))
        elif state.kind is SelectorKind.AREA:
            assert isinstance(state.value, RectangleRange)
            if isinstance(x_source, AxisRef):
                ranges.append((x_source, state.value.x))
            semantic = self._spec.cell if isinstance(self._spec, FacetGridPlot) else self._spec
            y_ref = getattr(semantic, "y", None)
            if isinstance(y_ref, AxisRef):
                ranges.append((y_ref, state.value.y))
        elif state.kind is SelectorKind.CROSSHAIR:
            if flat_indices.size != 1:
                return None
            physical = np.unravel_index(
                int(flat_indices[0]),
                schema.physical_shape,
            )
            semantic = (
                self._spec.cell
                if isinstance(self._spec, FacetGridPlot)
                else self._spec
            )
            for ref in (getattr(semantic, "x", None), getattr(semantic, "y", None)):
                if not isinstance(ref, AxisRef):
                    continue
                if ref.domain is AxisDomain.REPEAT:
                    terms.append(IndexSelection(schema.repeat_axis.axis_id, physical[0]))
                elif ref.domain is AxisDomain.DATA and ref.axis_id is not None:
                    try:
                        position = next(
                            index
                            for index, axis in enumerate(schema.cell_schema.data_axes)
                            if axis.axis_id.value == ref.axis_id
                        )
                    except StopIteration:
                        return None
                    terms.append(
                        IndexSelection(
                            AxisId(ref.axis_id),
                            int(physical[2 + position]),
                        )
                    )
                else:
                    # zlc_data.Selection intentionally names tensor axes only.
                    return None

        if not ranges and state.kind is not SelectorKind.CROSSHAIR:
            return None

        for ref, bounds in ranges:
            if ref.domain is AxisDomain.REPEAT:
                axis = schema.repeat_axis
            elif ref.domain is AxisDomain.DATA and ref.axis_id is not None:
                try:
                    axis = schema.cell_schema.axis(AxisId(ref.axis_id))
                except KeyError:
                    return None
            else:
                return None
            coordinates = np.asarray(
                tuple(axis.coordinate_at(index) for index in range(axis.size))
            )
            if coordinates.dtype.kind not in "biuf":
                return None
            selected = np.flatnonzero(
                (coordinates >= bounds.low) & (coordinates <= bounds.high)
            )
            if selected.size == 0 or not np.array_equal(
                selected,
                np.arange(selected[0], selected[-1] + 1),
            ):
                return None
            terms.append(
                IndexRangeSelection(
                    axis.axis_id,
                    int(selected[0]),
                    int(selected[-1]) + 1,
                )
            )
        if not terms:
            return None
        try:
            return Selection(tuple(terms))
        except ValueError:
            return None

    def _pulse_timeline_selector_data(
        self,
        kind: SelectorKind,
    ) -> PulseTimelineSelectionData:
        payload = self._projection.data
        if not isinstance(payload, PulseTimelineData):
            raise TypeError("PulseTimeline selector data requires PulseTimelineData")
        state = self._selector_controller.state(kind)
        display_state = self._special_display_selector_state(state)
        time_range: NumericRange | None
        if state.kind is SelectorKind.X_RANGE:
            assert isinstance(state.value, NumericRange)
            time_range = state.value
        elif state.kind is SelectorKind.AREA:
            assert isinstance(state.value, RectangleRange)
            time_range = state.value.x
        elif state.kind is SelectorKind.CROSSHAIR:
            assert isinstance(state.value, CrosshairPoint)
            time_range = NumericRange(state.value.x, state.value.x)
        else:
            time_range = None

        def intersects(start: float, stop: float) -> bool:
            return bool(
                time_range is not None
                and start <= time_range.high
                and stop >= time_range.low
            )

        blocks = tuple(
            record for record in payload.blocks if intersects(record.start, record.stop)
        )
        channel_ids = {record.channel_id for record in blocks}
        channels = tuple(
            record for record in payload.channels if record.channel_id in channel_ids
        )
        analog_traces = tuple(
            record
            for record in payload.analog_traces
            if record.starts and intersects(record.starts[0], record.starts[-1])
        )
        scan_regions = tuple(
            record
            for record in payload.scan_regions
            if intersects(record.start, record.stop)
        )
        scan_dac_segments = tuple(
            record
            for record in payload.scan_dac_segments
            if intersects(record.start, record.stop)
        )
        repeat_markers = tuple(
            record
            for record in payload.repeat_markers
            if intersects(record.start, record.stop)
        )
        return PulseTimelineSelectionData(
            state,
            display_state,
            channels,
            blocks,
            analog_traces,
            scan_regions,
            scan_dac_segments,
            repeat_markers,
            self.data_revision,
        )

    def _semantic_refs(self) -> tuple[AxisRef, ...]:
        semantic = self._spec.cell if isinstance(self._spec, FacetGridPlot) else self._spec
        result = []
        for ref in (
            getattr(semantic, "x", None),
            getattr(semantic, "y", None),
            getattr(semantic, "group", None),
            self._spec.facet if isinstance(self._spec, FacetGridPlot) else None,
        ):
            if isinstance(ref, AxisRef) and ref not in result:
                result.append(ref)
        return tuple(result)


    def _viewport_from_canonical(self, viewport: RectangleRange) -> RectangleRange:
        return RectangleRange(
            self._projected._canonical_range_to_display(
                viewport.x, self._projected._x_selector_source()
            ),
            viewport.y
            if self._projected._is_histogram_plot()
            else self._projected._canonical_range_to_display(
                viewport.y, self._projected._y_ref_or_value()
            ),
        )

    @property
    def viewport(self) -> RectangleRange | None:
        """Current explicit display-space zoom/pan region, if the user set one."""

        with self._lock:
            return self._viewport

    def _viewport_y_to_axes(self, value: NumericRange) -> tuple[float, float]:
        if isinstance(self._projected._semantic_spec(), ImagePlot):
            return value.high, value.low
        return value.low, value.high

    def set_viewport(
        self,
        x: NumericRange,
        y: NumericRange,
    ) -> RectangleRange:
        if not isinstance(x, NumericRange) or not isinstance(y, NumericRange):
            raise TypeError("viewport x and y must be NumericRange")
        selected = RectangleRange(x, y)
        self._set_viewport_state(selected)
        return selected

    def _set_viewport_state(self, selected: RectangleRange | None) -> bool:
        """Commit a display viewport without changing fit/data authority.

        A viewport is a presentation transform.  It must not clear a fit or
        arm a second solve while the pointer is moving; the accepted fit is
        recomputed only for a newer data revision or an explicit Fit command.
        """

        if selected is not None and not isinstance(selected, RectangleRange):
            raise TypeError("selected must be RectangleRange or None")
        with self._render_lock:
            with self._lock:
                self._assert_open()
                if selected == self._viewport:
                    return False
                previous = (
                    self._viewport,
                    self._fit_context_generation,
                )
                fit_cancel = self._fit_cancel
                self._viewport = selected
                self._fit_context_generation += 1
            effects = (
                RenderEffect.AXIS_TRANSFORM
                | RenderEffect.FIT_SELECTION
                | RenderEffect.OVERLAY
            )
            try:
                self._render_current(effects, schedule_fit=False)
            except BaseException:
                with self._lock:
                    (
                        self._viewport,
                        self._fit_context_generation,
                    ) = previous
                try:
                    self._render_current(effects, schedule_fit=False)
                except BaseException:
                    pass
                raise
            fit_cancel.set()
        return True

    def reset_viewport(self) -> None:
        self._set_viewport_state(None)

    def fit_selection(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None = None,
    ) -> FitSelection:
        """Freeze the exact painted samples that a fit would consume."""

        with self._render_lock:
            with self._lock:
                self._assert_open()
                resolved = self._resolve_fit_model(model)
                return self._projected.fit_selection(
                    resolved,
                    selector_kind=selector_kind,
                )

    def fit(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None = None,
        initial: Mapping[str, float] | Sequence[float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        options: FitOptions | None = None,
        cancelled: Callable[[], bool] | None = None,
        live: bool = True,
        fit_all_facets: bool = False,
    ) -> FitResult | FacetFitBatchResult:
        """Fit now; an all-facets request is one exact, one-shot batch."""

        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        if not isinstance(fit_all_facets, bool):
            raise TypeError("fit_all_facets must be bool")
        if fit_all_facets:
            started = self._begin_facet_fit_request(
                model,
                selector_kind=selector_kind,
                initial=initial,
                bounds=bounds,
                options=options,
            )
            result = self._solve_facet_fit_request(started, cancelled=cancelled)
            presentation = self._accept_facet_fit(result, started)
            if presentation is not None:
                self._notify_fit(presentation.event)
            return result
        started = self._begin_fit_request(
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
            live=live,
            logical_completion=None,
        )
        if started is None:
            raise RuntimeError("a synchronous fit request has no selection")
        result = self._solve_started_fit(started, cancelled=cancelled)
        if (
            not started.cancellation.is_set()
            and self.data_revision == started.selection.data_revision
        ):
            event = self._accept_fit(
                result,
                started,
            )
            if event is not None:
                self._notify_fit(event.event)
        return result

    def fit_async(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None = None,
        initial: Mapping[str, float] | Sequence[float] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        options: FitOptions | None = None,
        live: bool = True,
        fit_all_facets: bool = False,
    ) -> Future[FitResult | FacetFitBatchResult]:
        """Fit off-thread; all-facets is one exact, non-live batch."""

        if not isinstance(fit_all_facets, bool):
            raise TypeError("fit_all_facets must be bool")
        if fit_all_facets:
            completion: Future[FitResult | FacetFitBatchResult] = Future()
            try:
                started = self._begin_facet_fit_request(
                    model,
                    selector_kind=selector_kind,
                    initial=initial,
                    bounds=bounds,
                    options=options,
                )
                future = self._fit_executor.submit(
                    self._solve_facet_fit_request,
                    started,
                )
            except BaseException as error:
                completion.set_exception(error)
                return completion
            future.add_done_callback(
                lambda resolved: self._schedule_facet_fit_completion(
                    resolved,
                    started,
                    completion,
                )
            )
            return completion

        logical_completion: Future[FitResult] = Future()
        started = self._begin_fit_request(
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
            live=live,
            logical_completion=logical_completion,
        )
        if started is None:
            return logical_completion
        self._submit_started_fit(
            started,
            live=live,
            logical_completion=logical_completion,
        )
        return logical_completion

    def _begin_facet_fit_request(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        initial: Mapping[str, float] | Sequence[float] | None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
        options: FitOptions | None,
    ) -> _StartedFacetFitRequest:
        request = self._prepare_fit_request(
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
        )
        with self._render_lock:
            with self._lock:
                self._assert_open()
                if not isinstance(self._spec, FacetGridPlot):
                    raise TypeError("fit_all_facets requires FacetGridPlot")
                facet_values, selections = (
                    self._projected._all_facet_fit_selections(
                        request.model,
                        selector_kind=request.selector_kind,
                    )
                )
                facet = self._spec.facet
                previous_cancel = self._fit_cancel
                superseded = self._live_fit_completion
                self._fit_context_generation += 1
                self._fit_request_generation += 1
                cancellation = Event()
                self._fit_cancel = cancellation
                self._live_fit_completion = None
                self._live_fit_request = None
                self._live_fit_future = None
                self._live_fit_pending = False
                started = _StartedFacetFitRequest(
                    request,
                    facet,
                    facet_values,
                    selections,
                    cancellation,
                    self._fit_context_generation,
                    self._fit_request_generation,
                )
            previous_cancel.set()
        if superseded is not None and not superseded.done():
            superseded.set_exception(FitCancelled("fit request superseded"))
        return started

    def _solve_facet_fit_request(
        self,
        started: _StartedFacetFitRequest,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> FacetFitBatchResult:
        results: list[FitResult | None] = []
        errors: list[str | None] = []

        def should_cancel() -> bool:
            return started.cancellation.is_set() or (
                cancelled is not None and bool(cancelled())
            )

        for selection in started.selections:
            if should_cancel():
                raise FitCancelled("facet fit was cancelled")
            try:
                result = self._solve_fit_selection(
                    started.request.model,
                    selection,
                    initial=started.request.initial,
                    bounds=started.request.bounds,
                    options=started.request.options,
                    cancelled=should_cancel,
                )
            except FitCancelled:
                raise
            except (ArithmeticError, RuntimeError, ValueError) as error:
                results.append(None)
                errors.append(str(error).strip() or type(error).__name__)
            else:
                results.append(result)
                errors.append(None)
        return FacetFitBatchResult(
            started.facet,
            started.facet_values,
            started.request.model,
            tuple(results),
            tuple(errors),
            started.selections[0].data_revision,
        )

    def _begin_fit_request(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        initial: Mapping[str, float] | Sequence[float] | None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
        options: FitOptions | None,
        live: bool,
        logical_completion: Future[FitResult] | None,
    ) -> _StartedFitRequest | None:
        """Atomically replace fit request authority and freeze one solver input."""

        if not isinstance(live, bool):
            raise TypeError("live must be bool")
        request = self._prepare_fit_request(
            model,
            selector_kind=selector_kind,
            initial=initial,
            bounds=bounds,
            options=options,
        )
        selection: FitSelection | None = None
        with self._render_lock:
            with self._lock:
                self._assert_open()
                try:
                    selection = self._projected.fit_selection(
                        request.model,
                        selector_kind=request.selector_kind,
                    )
                except (TypeError, ValueError):
                    if logical_completion is None or not live:
                        raise
                previous_fit_cancel = self._fit_cancel
                self._fit_context_generation += 1
                self._fit_request_generation += 1
                superseded = self._live_fit_completion
                self._live_fit_completion = logical_completion if live else None
                cancellation = Event()
                self._fit_cancel = cancellation
                self._live_fit_request = request if live else None
                self._live_fit_pending = False
                started = (
                    None
                    if selection is None
                    else _StartedFitRequest(
                        request=request,
                        selection=selection,
                        cancellation=cancellation,
                        context_generation=self._fit_context_generation,
                        request_generation=self._fit_request_generation,
                    )
                )
            previous_fit_cancel.set()
        if superseded is not None and not superseded.done():
            superseded.set_exception(FitCancelled("fit request superseded"))
        return started

    def _prepare_fit_request(
        self,
        model: str | FitModelSpec,
        *,
        selector_kind: SelectorKind | None,
        initial: Mapping[str, float] | Sequence[float] | None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
        options: FitOptions | None,
    ) -> _LiveFitRequest:
        model_spec = self._resolve_fit_model(model)
        if options is not None and not isinstance(options, FitOptions):
            raise TypeError("options must be FitOptions or None")
        frozen_initial: Mapping[str, float] | tuple[float, ...] | None
        if initial is None:
            frozen_initial = None
        elif isinstance(initial, Mapping):
            frozen_initial = MappingProxyType(
                {str(name): float(value) for name, value in initial.items()}
            )
        elif isinstance(initial, (str, bytes)):
            raise TypeError("initial must be a parameter mapping or numeric sequence")
        else:
            try:
                frozen_initial = tuple(float(value) for value in initial)
            except TypeError as error:
                raise TypeError(
                    "initial must be a parameter mapping or numeric sequence"
                ) from error
        frozen_bounds = None
        if bounds is not None:
            if not isinstance(bounds, Mapping):
                raise TypeError("bounds must be a mapping or None")
            prepared_bounds: dict[str, tuple[float | None, float | None]] = {}
            for name, pair in bounds.items():
                try:
                    low, high = pair
                except (TypeError, ValueError) as error:
                    raise TypeError("each fit bound must contain low and high") from error
                prepared_bounds[str(name)] = (
                    None if low is None else float(low),
                    None if high is None else float(high),
                )
            frozen_bounds = MappingProxyType(prepared_bounds)
        if selector_kind is not None and not isinstance(selector_kind, SelectorKind):
            raise TypeError("selector_kind must be SelectorKind or None")
        return _LiveFitRequest(
            model_spec,
            selector_kind,
            frozen_initial,
            frozen_bounds,
            options,
        )

    def _solve_started_fit(
        self,
        started: _StartedFitRequest,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> FitResult:
        """Execute one frozen fit request on the caller or analysis worker."""

        def should_cancel() -> bool:
            return started.cancellation.is_set() or (
                cancelled is not None and bool(cancelled())
            )

        return self._solve_fit_selection(
            started.request.model,
            started.selection,
            initial=started.request.initial,
            bounds=started.request.bounds,
            options=started.request.options,
            cancelled=should_cancel,
        )

    def _submit_started_fit(
        self,
        started: _StartedFitRequest,
        *,
        live: bool,
        logical_completion: Future[FitResult] | None,
    ) -> Future[FitResult] | None:
        """Run one started fit on the analysis executor and route completion."""

        try:
            future = self._fit_executor.submit(self._solve_started_fit, started)
        except BaseException as error:
            if not live and logical_completion is not None:
                logical_completion.set_exception(error)
            # A later data or selector invalidation retries an armed live fit.
            return None
        tracked = False
        if live:
            with self._lock:
                tracked = (
                    not self._closed
                    and started.request_generation == self._fit_request_generation
                    and self._live_fit_request is started.request
                )
                if tracked:
                    self._live_fit_future = future
                else:
                    started.cancellation.set()
        future.add_done_callback(
            lambda completed: self._schedule_fit_completion(
                completed,
                started,
                logical_completion=None if live else logical_completion,
                tracked=tracked,
            )
        )
        return future

    def _solve_fit_selection(
        self,
        model: FitModelSpec,
        selection: FitSelection,
        *,
        initial: Mapping[str, float] | Sequence[float] | None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None,
        options: FitOptions | None,
        cancelled: Callable[[], bool] | None,
    ) -> FitResult:
        regular = selection.regular_image
        if regular is not None:
            result = self._fit_engine.fit(
                model,
                regular,
                data_revision=selection.data_revision,
                initial=initial,
                bounds=bounds,
                options=options,
                cancelled=cancelled,
            )
        else:
            result = self._fit_engine.fit(
                model,
                selection.coordinates,
                selection.observations,
                selected_indices=selection.selected_indices,
                data_revision=selection.data_revision,
                initial=initial,
                bounds=bounds,
                options=options,
                cancelled=cancelled,
            )
        if not isinstance(result, FitResult):
            raise TypeError("FitEngine.fit must return FitResult")
        if result.model != model:
            raise RuntimeError("FitEngine returned a result for another model")
        if result.data_revision != selection.data_revision:
            raise RuntimeError("FitEngine returned a result for another data revision")
        return result

    def _start_pending_live_fit(self) -> Future[FitResult] | None:
        """Launch the newest invalidated live fit; one active fit is sufficient."""

        with self._render_lock:
            with self._lock:
                if (
                    self._closed
                    or self._live_fit_request is None
                    or not self._live_fit_pending
                    or self._live_fit_future is not None
                ):
                    return self._live_fit_future
                request = self._live_fit_request
                self._live_fit_pending = False
                cancellation = Event()
                self._fit_cancel = cancellation
                try:
                    selection = self._projected.fit_selection(
                        request.model,
                        selector_kind=request.selector_kind,
                    )
                except (KeyError, TypeError, ValueError):
                    # A live revision may temporarily contain too few finite
                    # points.  Keep the fit mode armed; the next accepted data
                    # or selector revision will try the latest snapshot again.
                    return None
                started = _StartedFitRequest(
                    request,
                    selection,
                    cancellation,
                    self._fit_context_generation,
                    self._fit_request_generation,
                )

            return self._submit_started_fit(
                started,
                live=True,
                logical_completion=None,
            )

    def _schedule_fit_completion(
        self,
        future: Future[FitResult],
        started: _StartedFitRequest,
        *,
        logical_completion: Future[FitResult] | None,
        tracked: bool,
    ) -> None:
        resolutions: list[_FitResolution] = []
        presentations: list[_FitPresentation] = []
        restart_after_promotion = False

        def accept_and_paint() -> None:
            nonlocal restart_after_promotion
            try:
                resolution, presentation, restart = self._finish_fit_future(
                    future,
                    started,
                    logical_completion=logical_completion,
                    tracked=tracked,
                )
            except BaseException as error:
                if tracked:
                    failed = self._retire_failed_live_presentation(started, error)
                    if failed is not None:
                        resolutions.append(failed)
                raise
            if resolution is not None:
                resolutions.append(resolution)
            if presentation is not None:
                presentations.append(presentation)
            restart_after_promotion = restart_after_promotion or restart

        def finalize_presentation() -> None:
            for resolution in resolutions:
                self._resolve_fit_completion(resolution)
            for presentation in presentations:
                self._notify_fit(presentation.event)
            if restart_after_promotion:
                self._start_pending_live_fit()

        def abort_presentation() -> None:
            for presentation in reversed(presentations):
                self._abort_fit_presentation(presentation)

        try:
            with self._ownership_gate:
                with self._lock:
                    presentation_dispatch = self._presentation_dispatch
                if presentation_dispatch is None:
                    presented = self.owner_dispatch(accept_and_paint)
                else:
                    presented = presentation_dispatch(
                        accept_and_paint,
                        finalize_presentation,
                        abort_presentation,
                    )
                if not isinstance(presented, Future):
                    raise TypeError(
                        "host presentation dispatch must return "
                        "concurrent.futures.Future"
                    )
        except BaseException as error:
            targets = tuple(resolutions) or (
                ()
                if logical_completion is None
                else (_FitResolution(logical_completion, error=error),)
            )
            for resolution in targets:
                self._resolve_fit_completion(resolution)
            return

        def presentation_finished(completion: Future[Any]) -> None:
            try:
                completion.result()
            except BaseException as error:
                targets = tuple(resolutions) or (
                    ()
                    if logical_completion is None
                    else (_FitResolution(logical_completion, error=error),)
                )
                for resolution in targets:
                    target = resolution.completion
                    if not target.done():
                        target.set_exception(error)
            else:
                if presentation_dispatch is None:
                    finalize_presentation()

        presented.add_done_callback(presentation_finished)

    def _retire_failed_live_presentation(
        self,
        started: _StartedFitRequest,
        error: BaseException,
    ) -> _FitResolution | None:
        """Fail the first-result Future without disarming later live retries."""

        with self._lock:
            if (
                self._closed
                or self._live_fit_request is not started.request
                or self._fit_request_generation != started.request_generation
            ):
                return None
            self._live_fit_pending = True
            completion = self._live_fit_completion
            self._live_fit_completion = None
        return (
            None
            if completion is None
            else _FitResolution(completion, error=error)
        )

    @staticmethod
    def _resolve_fit_completion(resolution: _FitResolution) -> None:
        completion = resolution.completion
        if completion.done():
            return
        if resolution.error is not None:
            completion.set_exception(resolution.error)
        elif resolution.result is not None:
            completion.set_result(resolution.result)
        else:
            completion.set_exception(FitCancelled("fit did not reach presentation"))

    def _finish_fit_future(
        self,
        future: Future[FitResult],
        started: _StartedFitRequest,
        *,
        logical_completion: Future[FitResult] | None,
        tracked: bool,
    ) -> tuple[_FitResolution | None, _FitPresentation | None, bool]:
        current = False
        if tracked:
            with self._lock:
                current = self._live_fit_future is future
                if current:
                    self._live_fit_future = None
        result: FitResult | None = None
        error: BaseException | None = None
        if not future.cancelled():
            try:
                result = future.result()
            except FitCancelled as caught:
                error = caught
            except BaseException as caught:
                error = caught
                # A solver/data failure is transient for an armed live fit.
                # Leave the logical request pending; a later invalidation may
                # retry it, but this completion never schedules itself again.
        else:
            error = FitCancelled("fit analysis was cancelled")
        presentation: _FitPresentation | None = None
        if result is not None and not started.cancellation.is_set():
            presentation = self._accept_fit(
                result,
                started,
            )
        accepted = presentation is not None
        resolution: _FitResolution | None = None
        if tracked and current:
            with self._lock:
                request_current = (
                    not self._closed
                    and self._live_fit_request is not None
                    and started.request_generation == self._fit_request_generation
                )
                if request_current and accepted:
                    accepted_completion = self._live_fit_completion
                    self._live_fit_completion = None
                else:
                    accepted_completion = None
            if accepted_completion is not None and result is not None:
                resolution = _FitResolution(accepted_completion, result=result)
        elif logical_completion is not None:
            if error is not None:
                resolution = _FitResolution(logical_completion, error=error)
            elif accepted and result is not None:
                resolution = _FitResolution(logical_completion, result=result)
            else:
                resolution = _FitResolution(
                    logical_completion,
                    error=FitCancelled(
                        "fit result was superseded before presentation"
                    ),
                )
        return resolution, presentation, current

    def _schedule_facet_fit_completion(
        self,
        future: Future[FacetFitBatchResult],
        started: _StartedFacetFitRequest,
        completion: Future[FitResult | FacetFitBatchResult],
    ) -> None:
        presentation: _FacetFitPresentation | None = None
        result: FacetFitBatchResult | None = None

        def accept_and_paint() -> None:
            nonlocal presentation, result
            if future.cancelled():
                raise FitCancelled("facet fit was cancelled")
            result = future.result()
            if started.cancellation.is_set():
                raise FitCancelled("facet fit was superseded")
            presentation = self._accept_facet_fit(result, started)
            if presentation is None:
                raise FitCancelled(
                    "facet fit result was superseded before presentation"
                )

        def finalize_presentation() -> None:
            if result is None or presentation is None:
                raise RuntimeError("facet fit presentation was not prepared")
            if not completion.done():
                completion.set_result(result)
            self._notify_fit(presentation.event)

        def abort_presentation() -> None:
            if presentation is not None:
                self._abort_facet_fit_presentation(presentation)

        try:
            with self._ownership_gate:
                with self._lock:
                    presentation_dispatch = self._presentation_dispatch
                if presentation_dispatch is None:
                    presented = self.owner_dispatch(accept_and_paint)
                else:
                    presented = presentation_dispatch(
                        accept_and_paint,
                        finalize_presentation,
                        abort_presentation,
                    )
                if not isinstance(presented, Future):
                    raise TypeError(
                        "host presentation dispatch must return "
                        "concurrent.futures.Future"
                    )
        except BaseException as error:
            if not completion.done():
                completion.set_exception(error)
            return

        def presentation_finished(value: Future[Any]) -> None:
            try:
                value.result()
            except BaseException as error:
                if not completion.done():
                    completion.set_exception(error)
            else:
                if presentation_dispatch is None:
                    try:
                        finalize_presentation()
                    except BaseException as error:
                        if not completion.done():
                            completion.set_exception(error)

        presented.add_done_callback(presentation_finished)

    def _accept_facet_fit(
        self,
        result: FacetFitBatchResult,
        started: _StartedFacetFitRequest,
    ) -> _FacetFitPresentation | None:
        with self._render_lock:
            with self._lock:
                if (
                    self._closed
                    or result.source_revision != self.data_revision
                    or started.request_generation != self._fit_request_generation
                    or not isinstance(self._spec, FacetGridPlot)
                ):
                    return None
                overlays = tuple(
                    self._projected._make_fit_overlay(cell_result, selection)
                    for cell_result, selection in zip(
                        result.results,
                        started.selections,
                        strict=True,
                    )
                    if cell_result is not None
                )
                previous_fit = self._accepted_fit
                previous_batch = self._accepted_facet_fit
                accepted = _AcceptedFacetFit(
                    result,
                    started.selections,
                    overlays,
                    started.context_generation,
                )
                source = self._projection.data
                if not isinstance(source, DatasetSnapshot):
                    raise TypeError("Facet Fit requires a Dataset source")
                parameter_units = self._projected._fit_parameter_canonical_units(
                    result.model
                )
                self._accepted_fit = None
                self._accepted_facet_fit = accepted
            try:
                self._render_current(
                    RenderEffect.OVERLAY,
                    schedule_fit=False,
                )
            except BaseException:
                with self._lock:
                    if self._accepted_facet_fit is accepted:
                        self._accepted_fit = previous_fit
                        self._accepted_facet_fit = previous_batch
                try:
                    self._render_current(
                        RenderEffect.OVERLAY,
                        schedule_fit=False,
                    )
                except BaseException:
                    pass
                raise
        return _FacetFitPresentation(
            FitEvent(
                result,
                started.selections,
                (),
                result.model.formula or "",
                source.source,
                parameter_units,
            ),
            accepted,
            previous_fit,
            previous_batch,
        )

    def _abort_facet_fit_presentation(
        self,
        presentation: _FacetFitPresentation,
    ) -> None:
        with self._render_lock:
            with self._lock:
                if self._accepted_facet_fit is not presentation.accepted:
                    raise RuntimeError("facet fit presentation is no longer current")
                self._accepted_fit = presentation.previous_fit
                self._accepted_facet_fit = presentation.previous_batch
            self._render_current(
                RenderEffect.OVERLAY,
                schedule_fit=False,
            )

    def _accept_fit(
        self,
        result: FitResult,
        started: _StartedFitRequest,
    ) -> _FitPresentation | None:
        selection = started.selection
        with self._render_lock:
            with self._lock:
                if (
                    self._closed
                    or result.data_revision != self.data_revision
                    or started.request_generation != self._fit_request_generation
                ):
                    return None
                if selection.data_revision != self.data_revision:
                    return None
                overlay = self._projected._make_fit_overlay(result, selection)
                previous = self._accepted_fit
                previous_batch = self._accepted_facet_fit
                accepted = _AcceptedFit(
                    result=result,
                    selection=selection,
                    overlay=overlay,
                    context_generation=started.context_generation,
                )
                source = self._projection.data
                if not isinstance(source, DatasetSnapshot):
                    raise TypeError("Fit requires a Dataset source")
                parameter_units = self._projected._fit_parameter_canonical_units(
                    result.model
                )
                self._accepted_fit = accepted
                self._accepted_facet_fit = None
            try:
                self._render_current(
                    RenderEffect.OVERLAY,
                    schedule_fit=False,
                )
            except BaseException:
                with self._lock:
                    if self._accepted_fit is accepted:
                        self._accepted_fit = previous
                        self._accepted_facet_fit = previous_batch
                try:
                    self._render_current(
                        RenderEffect.OVERLAY,
                        schedule_fit=False,
                    )
                except BaseException:
                    pass
                raise
        return _FitPresentation(
            FitEvent(
                result,
                selection,
                overlay.parameter_display,
                overlay.formula,
                source.source,
                parameter_units,
            ),
            accepted,
            previous,
            previous_batch,
        )

    def _abort_fit_presentation(self, presentation: _FitPresentation) -> None:
        """Restore the accepted fit preceding an unpromoted raster front."""

        if not isinstance(presentation, _FitPresentation):
            raise TypeError("presentation must be a fit presentation")
        with self._render_lock:
            with self._lock:
                if self._accepted_fit is not presentation.accepted:
                    raise RuntimeError("fit presentation is no longer current")
                self._accepted_fit = presentation.previous
                self._accepted_facet_fit = presentation.previous_batch
            self._render_current(
                RenderEffect.OVERLAY,
                schedule_fit=False,
            )

    def subscribe_fit(self, callback: FitCallback) -> Callable[[], None]:
        """Observe results only after they are accepted and painted."""

        return self._subscribe_callback(self._fit_callbacks, callback)

    def _notify_fit(self, event: FitEvent) -> None:
        with self._lock:
            callbacks = tuple(self._fit_callbacks)
        self._notify_callbacks(callbacks, event)


    def clear_fit(self) -> None:
        logical_completion: Future[FitResult] | None = None
        with self._render_lock:
            with self._lock:
                self._assert_open()
                logical_completion = self._live_fit_completion
                previous = (
                    self._live_fit_request,
                    self._live_fit_pending,
                    self._live_fit_completion,
                    self._fit_request_generation,
                    self._fit_context_generation,
                    self._accepted_fit,
                    self._accepted_facet_fit,
                )
                fit_cancel = self._fit_cancel
                self._live_fit_completion = None
                self._fit_request_generation += 1
                self._live_fit_request = None
                self._live_fit_pending = False
                self._fit_context_generation += 1
                self._accepted_fit = None
                self._accepted_facet_fit = None
            try:
                self._render_current(
                    RenderEffect.OVERLAY,
                    schedule_fit=False,
                )
            except BaseException:
                with self._lock:
                    (
                        self._live_fit_request,
                        self._live_fit_pending,
                        self._live_fit_completion,
                        self._fit_request_generation,
                        self._fit_context_generation,
                        self._accepted_fit,
                        self._accepted_facet_fit,
                    ) = previous
                try:
                    self._render_current(
                        RenderEffect.OVERLAY,
                        schedule_fit=False,
                    )
                except BaseException:
                    pass
                raise
            fit_cancel.set()
        if logical_completion is not None and not logical_completion.done():
            logical_completion.set_exception(FitCancelled("fit request cleared"))


    def _pulse_display_x_to_source(self, value: float) -> float:
        return float(value) / self._projected._pulse_x_factor()

    def _pulse_source_x_to_display(self, value: float) -> float:
        return float(value) * self._projected._pulse_x_factor()

    def _pulse_display_range_to_source(
        self, value: NumericRange
    ) -> NumericRange:
        factor = self._projected._pulse_x_factor()
        return NumericRange(value.low / factor, value.high / factor)


    def _viewport_x_to_axes(self, value: NumericRange) -> NumericRange:
        return (
            self._pulse_display_range_to_source(value)
            if isinstance(self._spec, PulseTimelinePlot)
            else value
        )

    def _viewport_x_from_axes(self, value: NumericRange) -> NumericRange:
        return (
            self._projected._pulse_source_range_to_display(value)
            if isinstance(self._spec, PulseTimelinePlot)
            else value
        )

    def _display_x_scalar_to_canonical(self, value: float) -> float:
        if isinstance(self._spec, RollingPlot):
            canonical, display = self._projected._rolling_axis_domains()
            return float(np.interp(value, display, canonical))
        source = self._projected._x_selector_source()
        quantity = self._projected._coordinate(source) if isinstance(source, AxisRef) else source
        return self._projected._display_scalar_to_canonical(value, quantity)


    def _area_display_to_canonical(
        self,
        value: RectangleRange,
    ) -> RectangleRange:
        if self._view is not None:
            x = self._projected._display_range_to_canonical(
                value.x,
                self._projected._x_selector_source(),
            )
            y = (
                value.y
                if self._projected._is_histogram_plot()
                else self._projected._display_range_to_canonical(
                    value.y,
                    self._projected._y_ref_or_value(),
                )
            )
            return RectangleRange(x, y)
        if isinstance(self._spec, PulseTimelinePlot):
            return RectangleRange(
                self._pulse_display_range_to_source(value.x),
                value.y,
            )
        return value


    def _special_display_selector_state(self, state: SelectorState) -> SelectorState:
        if not isinstance(self._spec, PulseTimelinePlot):
            return state
        value = state.value
        if state.kind is SelectorKind.X_RANGE:
            assert isinstance(value, NumericRange)
            value = self._projected._pulse_source_range_to_display(value)
        elif state.kind is SelectorKind.AREA:
            assert isinstance(value, RectangleRange)
            value = self._projected._area_canonical_to_display(value)
        elif state.kind is SelectorKind.CROSSHAIR:
            assert isinstance(value, CrosshairPoint)
            value = CrosshairPoint(
                self._pulse_source_x_to_display(value.x), value.y
            )
        return replace(state, value=value)

    def _derived_threshold_selector(self) -> SelectorState | None:
        """Resolve the fit suggestion as a display selector fallback."""

        accepted = self._accepted_fit
        if accepted is None or not self._projected._is_histogram_plot():
            return None
        threshold = accepted.overlay.suggested_threshold
        if threshold is None:
            return None
        canonical = (
            self._display_x_scalar_to_canonical(threshold)
            if self._view is not None
            else float(threshold)
        )
        return SelectorState(
            SelectorKind.THRESHOLD,
            canonical,
            revision=accepted.context_generation,
            facet_index=accepted.selection.facet_index,
        )

    def _resolved_selector_snapshot(self) -> SelectorSnapshot:
        """Return the sole painted/hit-tested state for every selector kind."""

        snapshot = self._selector_controller.snapshot()
        if any(state.kind is SelectorKind.THRESHOLD for state in snapshot.states):
            return snapshot
        derived = self._derived_threshold_selector()
        if derived is None:
            return snapshot
        return SelectorSnapshot(snapshot.committed + (derived,), snapshot.candidate)

    def _display_selector_snapshot(self) -> SelectorSnapshot:
        snapshot = self._resolved_selector_snapshot()

        def display(state: SelectorState) -> SelectorState:
            return (
                self._projected._display_selector_state(state)
                if self._view is not None
                else self._special_display_selector_state(state)
            )

        return SelectorSnapshot(
            tuple(display(state) for state in snapshot.committed),
            None if snapshot.candidate is None else display(snapshot.candidate),
        )

    def _display_selector_states(self) -> tuple[SelectorState, ...]:
        return self._display_selector_snapshot().states

    def _raster_pointer_state(
        self,
        *,
        publish_front: bool,
    ) -> _PointerUpdate:
        """Return the transient pointer state needed by a raster frontend."""

        snapshot = self._display_selector_snapshot()
        color_candidate = self._display_color_limit_candidate()
        candidate: SelectorState | ColorLimitCandidate | None = (
            color_candidate if color_candidate is not None else snapshot.candidate
        )
        scene = (
            None
            if candidate is None or self._renderer is None
            else self._renderer.selector_scene(
                snapshot,
                color_candidate=color_candidate,
            )
        )
        gesture = self._gesture
        active_pan = isinstance(gesture, _PanGesture)
        axis = (
            gesture.axes
            if gesture is not None and (candidate is not None or active_pan)
            else None
        )
        if axis is None:
            return _PointerUpdate(
                candidate,
                scene,
                None,
                None,
                active_pan,
                publish_front,
            )
        role, separator, suffix = str(axis.get_gid() or "main").partition(":")
        cell_index = int(suffix) if separator and suffix.isdigit() else None
        return _PointerUpdate(
            candidate,
            scene,
            role or "main",
            cell_index,
            active_pan,
            publish_front,
        )

    def _raster_pointer_event(
        self,
        action: str,
        x: float,
        y: float,
        *,
        button: int | None = None,
        double: bool = False,
        step: float = 0.0,
        key: str | None = None,
        axes_snapshot: AxisTransform | None = None,
    ) -> _PointerUpdate:
        """Route normalized raster input through the native session handlers."""

        from matplotlib.backend_bases import KeyEvent, MouseEvent

        if not isinstance(action, str):
            raise TypeError("raster pointer action must be text")
        selected_action = action.strip().lower()
        if selected_action not in {
            "press",
            "move",
            "release",
            "scroll",
            "key",
            "cancel",
        }:
            raise ValueError(f"unknown raster pointer action {action!r}")
        coordinates = np.asarray((x, y), dtype=float)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("raster pointer coordinates must be finite")
        if button is not None and button not in (1, 2, 3):
            raise ValueError("raster pointer button must be 1, 2, 3, or None")
        if axes_snapshot is not None and not isinstance(
            axes_snapshot,
            AxisTransform,
        ):
            raise TypeError("axes_snapshot must be AxisTransform or None")
        with self._lock:
            self._assert_open()
        presentation_epoch = self._presentation_epoch
        assert self._renderer is not None
        canvas = self._renderer.figure.canvas
        width, height = canvas_physical_size(canvas)
        pixel_x = float(x) * float(width)
        pixel_y = (1.0 - float(y)) * float(height)
        interaction_transform = axes_snapshot
        if selected_action == "cancel":
            self.cancel_interaction()
        elif selected_action == "key":
            self._on_key_press(
                KeyEvent(
                    "key_press_event",
                    canvas,
                    key="" if key is None else str(key),
                    x=pixel_x,
                    y=pixel_y,
                )
            )
        elif selected_action == "scroll":
            if float(step) == 0.0:
                return self._raster_pointer_state(publish_front=False)
            direction = "up" if float(step) > 0.0 else "down"
            event = MouseEvent(
                "scroll_event",
                canvas,
                pixel_x,
                pixel_y,
                button=direction,
                step=float(step),
            )
            if interaction_transform is not None:
                event.inaxes = self._axis_for_transform(
                    interaction_transform
                )
            self._on_scroll(
                event,
                interaction_transform=interaction_transform,
            )
        else:
            event = MouseEvent(
                f"button_{selected_action}_event"
                if selected_action != "move"
                else "motion_notify_event",
                canvas,
                pixel_x,
                pixel_y,
                button=button,
                dblclick=bool(double),
            )
            if selected_action == "press":
                self._on_button_press(
                    event,
                    interaction_transform=interaction_transform,
                    external_selector_scene=True,
                )
            else:
                {
                    "move": self._on_motion,
                    "release": self._on_button_release,
                }[selected_action](event)
        return self._raster_pointer_state(
            publish_front=self._presentation_epoch != presentation_epoch,
        )

    @staticmethod
    def _event_coordinates(event: Any, axes: Any) -> tuple[float, float] | None:
        if (
            getattr(event, "inaxes", None) is axes
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            return float(event.xdata), float(event.ydata)
        x = getattr(event, "x", None)
        y = getattr(event, "y", None)
        if x is None or y is None:
            return None
        try:
            display_x = float(x)
            display_y = float(y)
            if not np.all(np.isfinite((display_x, display_y))):
                return None
            data_x, data_y = axes.transData.inverted().transform(
                (display_x, display_y)
            )
            if not np.all(np.isfinite((data_x, data_y))):
                return None
            return float(data_x), float(data_y)
        except (AttributeError, TypeError, ValueError):
            return None

    def _pointer_state(self) -> SelectorState | None:
        gesture = self._gesture
        if not isinstance(gesture, _SelectorGesture):
            return None
        pointer = self._selector_controller.candidate_state()
        return (
            pointer
            if pointer is not None and pointer.kind is gesture.kind
            else None
        )

    def _pointer_threshold_active(self) -> bool:
        pointer = self._pointer_state()
        return bool(
            pointer is not None
            and self._projected._is_histogram_plot()
            and pointer.kind is SelectorKind.THRESHOLD
        )

    def _event_canonical(
        self,
        event: Any,
        *,
        captured_transform: AxisTransform | None = None,
    ) -> CrosshairPoint | None:
        assert self._renderer is not None
        if captured_transform is not None:
            point = captured_transform.canonical_point(
                event,
                self._renderer.figure.canvas,
            )
            if point is None:
                return None
            return (
                CrosshairPoint(point.x, point.x)
                if self._pointer_threshold_active()
                else point
            )
        axes = getattr(event, "inaxes", None)
        if axes is None:
            return None
        coordinates = self._event_coordinates(event, axes)
        if coordinates is None:
            return None
        xdata, ydata = coordinates
        distribution_axes = self._renderer.axes.get("distribution", ())
        if axes in distribution_axes:
            value = ydata
            if self._view is not None:
                value = self._projected._display_scalar_to_canonical(
                    value,
                    self._projected._value_quantity(),
                )
            return CrosshairPoint(value, 0.0)
        if axes is not self._renderer.primary_axes:
            return None
        if self._view is None:
            return CrosshairPoint(xdata, ydata)
        point = CrosshairPoint(
            self._display_x_scalar_to_canonical(xdata),
            ydata
            if self._projected._is_histogram_plot()
            else self._projected._display_scalar_to_canonical(
                ydata, self._projected._y_ref_or_value()
            ),
        )
        return (
            CrosshairPoint(point.x, point.x)
            if self._pointer_threshold_active()
            else point
        )

    def _is_double_click(self, event: Any, button: int) -> bool:
        """Normalize native, browser-detail, and timed middle/right clicks."""

        explicit = bool(getattr(event, "dblclick", False))
        gui_event = getattr(event, "guiEvent", None)
        detail = getattr(gui_event, "detail", 0)
        try:
            browser_double = int(detail) >= 2
        except (TypeError, ValueError):
            browser_double = False
        if explicit or browser_double:
            self._click_history.pop(button, None)
            return True
        if button not in (2, 3):
            return False
        try:
            x = float(getattr(event, "x"))
            y = float(getattr(event, "y"))
        except (TypeError, ValueError):
            self._click_history.pop(button, None)
            return False
        if not np.all(np.isfinite((x, y))):
            self._click_history.pop(button, None)
            return False
        now = monotonic()
        previous = self._click_history.get(button)
        interval = self._defaults.interaction.double_click_interval_ms / 1000.0
        radius = self._defaults.interaction.double_click_radius_px
        if previous is not None:
            then, previous_x, previous_y = previous
            if now - then <= interval and math.hypot(
                x - previous_x,
                y - previous_y,
            ) <= radius:
                self._click_history.pop(button, None)
                return True
        self._click_history[button] = (now, x, y)
        return False

    def _on_button_press(
        self,
        event: Any,
        *,
        interaction_transform: AxisTransform | None = None,
        external_selector_scene: bool = False,
    ) -> None:
        button = getattr(event, "button", None)
        if button not in (1, 2, 3):
            return
        button = int(button)
        assert self._renderer is not None
        event_axes = self._renderer.interactive_axes_at(event)
        if interaction_transform is not None:
            event_axes = self._axis_for_transform(interaction_transform)
            event.inaxes = event_axes
        elif event_axes is not None:
            interaction_transform = self._axis_transform_for_axis(event_axes)
        is_double = self._is_double_click(event, button)
        self._cancel_gesture()
        if isinstance(self._spec, FacetGridPlot):
            focus_index = self._facet_focus_index
            if focus_index is None:
                if button == 1 and is_double:
                    facet_index = self._renderer.facet_index_for_axes(event_axes)
                    if facet_index is not None:
                        self.focus_facet(facet_index)
                return
            if button == 1 and is_double and event_axes is self._renderer.primary_axes:
                self.show_facet_overview()
                return
        if button == 2:
            if event_axes is not self._renderer.primary_axes:
                return
            if is_double:
                self.reset_viewport()
                return
            if interaction_transform is None:
                return
            origin = interaction_transform.display_point(
                event,
                self._renderer.figure.canvas,
            )
            if origin is None:
                return
            self._gesture = _PanGesture(
                event_axes,
                interaction_transform,
                origin,
                NumericRange(*interaction_transform.x_limits),
                NumericRange(*interaction_transform.y_limits),
            )
            return
        point = self._event_canonical(
            event,
            captured_transform=interaction_transform,
        )
        if point is None:
            return
        if button == 3:
            if event_axes is not self._renderer.primary_axes:
                return
            self._update_pointer_crosshair(
                point,
                clear=is_double,
            )
            return

        if event_axes in self._renderer.axes.get("distribution", ()):
            if interaction_transform is None:
                return
            self._begin_color_limit_pointer(
                point,
                event_axes,
                interaction_transform,
                external_selector_scene=external_selector_scene,
            )
            return
        hit = self._hit_selector(
            point,
            event_axes,
            event=event,
            transform=interaction_transform,
        )
        if hit is None:
            state = self._start_pointer_selection(point, event_axes)
            if state is None:
                return
            handle = DragHandle.NEW
        else:
            state, handle = hit
        selector_axes = self._selector_axes(state)
        if selector_axes is None or interaction_transform is None:
            return
        try:
            self._selector_controller.state(state.kind)
        except KeyError:
            draft = state
        else:
            draft = None
        self._selector_controller.pointer_down(
            state.kind,
            point.x,
            point.y,
            handle=handle,
            draft=draft,
        )
        self._gesture = _SelectorGesture(
            selector_axes,
            interaction_transform,
            state.kind,
            external_selector_scene,
        )
        if external_selector_scene:
            self._renderer.begin_external_selector_gesture(
                state.kind,
            )
            self._render_current(RenderEffect.OVERLAY)
        elif hit is None:
            self._render_current(RenderEffect.OVERLAY)
        if external_selector_scene:
            return
        with self._renderer.raster_transaction():
            self._renderer.begin_selector_gesture(state.kind)


    def _start_pointer_selection(
        self,
        point: CrosshairPoint,
        event_axes: Any,
    ) -> SelectorState | None:
        assert self._renderer is not None
        if event_axes is not self._renderer.primary_axes:
            return None
        value: SelectorValue = RectangleRange(
            NumericRange(point.x, point.x),
            NumericRange(point.y, point.y),
        )
        return SelectorState(
            SelectorKind.AREA,
            value,
            facet_index=self._focused_facet_index,
        )

    def _update_pointer_crosshair(
        self,
        point: CrosshairPoint,
        *,
        clear: bool,
    ) -> None:
        state = self._projected._selector_state_or_none(SelectorKind.CROSSHAIR)
        if clear:
            if state is not None:
                self.remove_selector(SelectorKind.CROSSHAIR)
            return
        self._install_selector_state(SelectorState(
            SelectorKind.CROSSHAIR,
            point,
            facet_index=self._focused_facet_index,
        ))

    def _begin_color_limit_pointer(
        self,
        point: CrosshairPoint,
        event_axes: Any,
        transform: AxisTransform,
        *,
        external_selector_scene: bool,
    ) -> bool:
        assert self._renderer is not None
        if not isinstance(self._spec, ImagePlot):
            return False
        distribution = self._renderer.axes.get("distribution", ())
        if event_axes not in distribution:
            return False
        bounds = self._color_selector_domain()
        value = self._current_color_limit_value()
        tolerance = max(
            bounds.span * self._defaults.interaction.selector_hit_radius_fraction,
            1.0e-12,
        )
        hit = _range_endpoint_hit(value, point.x, tolerance)
        if hit is None:
            return True
        _score, handle = hit
        minimum_span = min(
            bounds.span,
            max(value.span * 1.0e-12, bounds.span * 1.0e-12, 1.0e-12),
        )
        self._gesture = _ColorGesture(
            event_axes,
            transform,
            _ColorLimitDrag(
                original=value,
                candidate=value,
                handle=handle,
                origin=point.x,
                bounds=bounds,
                minimum_span=minimum_span,
            ),
            external_selector_scene,
        )
        candidate = self._display_color_limit_candidate()
        assert candidate is not None
        if external_selector_scene:
            self._renderer.begin_external_selector_gesture(
                SelectorSceneKind.COLOR_LIMITS,
            )
            self._render_current(RenderEffect.OVERLAY, schedule_fit=False)
        else:
            with self._renderer.raster_transaction():
                self._renderer.begin_color_limit_gesture(candidate)
        return True

    def _area_drag_handle(
        self,
        state: SelectorState,
        axes: Any,
        event: Any | None,
        transform: AxisTransform | None = None,
    ) -> tuple[float, DragHandle] | None:
        pixel_x = None if event is None else getattr(event, "x", None)
        pixel_y = None if event is None else getattr(event, "y", None)
        if pixel_x is None or pixel_y is None:
            return None
        displayed = (
            self._projected._display_selector_state(state)
            if self._view is not None
            else state
        )
        if not isinstance(displayed.value, RectangleRange):
            return None
        value = displayed.value
        x0, x1 = value.x.low, value.x.high
        y0, y1 = value.y.low, value.y.high
        xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        mouse = np.asarray((float(pixel_x), float(pixel_y)), dtype=float)

        def distance(coordinate: tuple[float, float]) -> float:
            shown = np.asarray(
                axes.transData.transform(coordinate)
                if transform is None
                else transform.display_to_pixel(
                    coordinate[0],
                    coordinate[1],
                    axes.figure.canvas,
                ),
                dtype=float,
            )
            return float(np.linalg.norm(shown - mouse))

        pixel_ratio = self.surface_plan.device_pixel_ratio
        center_radius = (
            self._defaults.interaction.selector_center_radius_px * pixel_ratio
        )
        center_distance = distance((xm, ym))
        handle_radius = (
            self._defaults.interaction.selector_handle_radius_px * pixel_ratio
        )
        handles = (
            ((x0, y0), DragHandle.BOTTOM_LEFT),
            ((xm, y0), DragHandle.BOTTOM),
            ((x1, y0), DragHandle.BOTTOM_RIGHT),
            ((x1, ym), DragHandle.RIGHT),
            ((x1, y1), DragHandle.TOP_RIGHT),
            ((xm, y1), DragHandle.TOP),
            ((x0, y1), DragHandle.TOP_LEFT),
            ((x0, ym), DragHandle.LEFT),
        )
        candidates = (
            (center_distance / center_radius, DragHandle.BODY),
            *(
                (distance(coordinate) / handle_radius, handle)
                for coordinate, handle in handles
            ),
        )
        score, nearest_handle = min(candidates, key=lambda item: item[0])
        return (score, nearest_handle) if score <= 1.0 else None

    def _on_scroll(
        self,
        event: Any,
        *,
        interaction_transform: AxisTransform | None = None,
    ) -> None:
        direction = getattr(event, "button", None)
        if direction not in ("up", "down"):
            return
        self._cancel_gesture()
        assert self._renderer is not None
        if (
            isinstance(self._spec, FacetGridPlot)
            and self._facet_focus_index is None
        ):
            return
        axes = (
            self._axis_for_transform(interaction_transform)
            if interaction_transform is not None
            else self._renderer.interactive_axes_at(event)
        )
        if axes is not self._renderer.primary_axes:
            return
        point = (
            interaction_transform.display_point(
                event,
                self._renderer.figure.canvas,
            )
            if interaction_transform is not None
            else None
        )
        coordinates = (
            None if point is None else (point.x, point.y)
        ) if interaction_transform is not None else self._event_coordinates(event, axes)
        if coordinates is None:
            return
        center_x, center_y = coordinates
        zoom_factor = self._defaults.interaction.wheel_zoom_factor
        factor = zoom_factor if direction == "up" else 1.0 / zoom_factor
        x_axes = NumericRange(
            *(
                interaction_transform.x_limits
                if interaction_transform is not None
                else tuple(map(float, axes.get_xlim()))
            )
        )
        y_axes = NumericRange(
            *(
                interaction_transform.y_limits
                if interaction_transform is not None
                else tuple(map(float, axes.get_ylim()))
            )
        )
        zoomed_x = NumericRange(
            center_x + (x_axes.low - center_x) * factor,
            center_x + (x_axes.high - center_x) * factor,
        )
        zoomed_y = y_axes
        if isinstance(self._projected._semantic_spec(), ImagePlot):
            zoomed_y = NumericRange(
                center_y + (y_axes.low - center_y) * factor,
                center_y + (y_axes.high - center_y) * factor,
            )
        self.set_viewport(self._viewport_x_from_axes(zoomed_x), zoomed_y)

    def _hit_selector(
        self,
        point: CrosshairPoint,
        event_axes: Any,
        *,
        event: Any | None = None,
        transform: AxisTransform | None = None,
    ) -> tuple[SelectorState, DragHandle] | None:
        x_bounds = self._selector_x_bounds(transform)
        y_bounds = self._selector_y_bounds(transform)
        hit_fraction = self._defaults.interaction.selector_hit_radius_fraction
        tx = max(x_bounds.span * hit_fraction, 1e-12)
        ty = max(y_bounds.span * hit_fraction, 1e-12)
        candidates: list[
            tuple[int, float, int, SelectorState, DragHandle]
        ] = []
        for order, state in enumerate(self._resolved_selector_snapshot().states):
            if self._selector_axes(state) is not event_axes:
                continue
            if state.facet_index != self._focused_facet_index:
                continue
            value = state.value
            score = float("inf")
            handle = DragHandle.BODY
            if state.kind is SelectorKind.X_RANGE and isinstance(value, NumericRange):
                endpoint = _range_endpoint_hit(value, point.x, tx)
                if endpoint is not None:
                    score, handle = endpoint
                elif value.low <= point.x <= value.high:
                    score, handle = 0.9, DragHandle.BODY
            elif state.kind is SelectorKind.AREA and isinstance(value, RectangleRange):
                area_hit = self._area_drag_handle(
                    state,
                    event_axes,
                    event,
                    transform,
                )
                if area_hit is not None:
                    score, handle = area_hit
            elif state.kind is SelectorKind.THRESHOLD:
                score = (
                    abs(point.x - float(value)) / tx
                    if self._projected._is_histogram_plot()
                    else abs(point.y - float(value)) / ty
                )
            if score <= 1.0:
                kind_priority = (
                    0 if state.kind is SelectorKind.THRESHOLD else 1
                )
                candidates.append(
                    (
                        kind_priority,
                        score,
                        order,
                        state,
                        handle,
                    )
                )
        if not candidates:
            return None
        _priority, _score, _order, state, handle = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        return state, handle

    def _coordinate_bounds(self, ref: AxisRef) -> NumericRange:
        values = np.asarray(self._projected._coordinate(ref).canonical, dtype=float)
        finite = values[np.isfinite(values)]
        return NumericRange(float(np.min(finite)), float(np.max(finite)))

    def _value_bounds(self) -> NumericRange:
        values = np.asarray(self._projected._value_quantity().canonical, dtype=float)
        finite = values[np.isfinite(values)]
        return NumericRange(float(np.min(finite)), float(np.max(finite)))

    def _selector_x_bounds(
        self,
        transform: AxisTransform | None = None,
    ) -> NumericRange:
        if transform is not None:
            return NumericRange(*transform.canonical_x_limits)
        if self._view is None:
            assert self._renderer is not None
            return NumericRange(
                *sorted(map(float, self._renderer.primary_axes.get_xlim()))
            )
        source = self._projected._x_selector_source()
        return (
            self._coordinate_bounds(source)
            if isinstance(source, AxisRef)
            else self._value_bounds()
        )

    def _color_selector_domain(self) -> NumericRange:
        assert self._renderer is not None
        axes = self._renderer.axes.get("distribution", ())
        if not axes:
            raise RuntimeError("color selector requires a side distribution")
        low, high = sorted(map(float, axes[0].get_ylim()))
        if self._view is not None:
            quantity = self._projected._value_quantity()
            low = self._projected._display_scalar_to_canonical(low, quantity)
            high = self._projected._display_scalar_to_canonical(high, quantity)
        return NumericRange(*sorted((low, high)))

    def _current_color_limit_value(self) -> NumericRange:
        assert self._renderer is not None
        low, high = sorted(self._renderer.resolved_color_limits())
        if self._view is not None:
            quantity = self._projected._value_quantity()
            low = self._projected._display_scalar_to_canonical(low, quantity)
            high = self._projected._display_scalar_to_canonical(high, quantity)
        return NumericRange(*sorted((low, high)))

    def _display_color_limit_candidate(self) -> ColorLimitCandidate | None:
        gesture = self._gesture
        if not isinstance(gesture, _ColorGesture):
            return None
        value = gesture.drag.candidate
        if self._view is not None:
            value = self._projected._canonical_range_to_display(
                value,
                self._projected._value_quantity(),
            )
        return ColorLimitCandidate(value)

    def _selector_y_bounds(
        self,
        transform: AxisTransform | None = None,
    ) -> NumericRange:
        if transform is not None:
            if self._projected._is_histogram_plot():
                pointer = self._pointer_state()
                if pointer is not None and pointer.kind is SelectorKind.THRESHOLD:
                    return NumericRange(*transform.canonical_x_limits)
            return NumericRange(*transform.canonical_y_limits)
        pointer = self._pointer_state()
        if (
            pointer is not None
            and self._projected._is_histogram_plot()
            and pointer.kind is SelectorKind.THRESHOLD
        ):
            return self._selector_x_bounds()
        if self._view is None:
            assert self._renderer is not None
            return NumericRange(
                *sorted(map(float, self._renderer.primary_axes.get_ylim()))
            )
        assert self._renderer is not None
        visible = NumericRange(
            *sorted(map(float, self._renderer.primary_axes.get_ylim()))
        )
        if self._projected._is_histogram_plot():
            return visible
        source = self._projected._y_ref_or_value()
        return self._projected._display_range_to_canonical(visible, source)

    def _on_motion(self, event: Any) -> None:
        gesture = self._gesture
        if gesture is None:
            return
        if isinstance(gesture, _PanGesture):
            if gesture.lane_due(
                "pan",
                self._defaults.interaction.pointer_update_interval_ms,
            ):
                self._update_pan(event, gesture)
            return
        point = self._event_canonical(
            event,
            captured_transform=gesture.transform,
        )
        if point is None:
            self._cancel_gesture()
            return
        if isinstance(gesture, _ColorGesture):
            gesture.drag = gesture.drag.moved(point.x)
            candidate = self._display_color_limit_candidate()
            assert candidate is not None
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                if not gesture.external_scene:
                    self._renderer.preview_color_limit_candidate(candidate)
                if gesture.lane_due(
                    "raster",
                    self._defaults.interaction.raster_preview_interval_ms,
                ):
                    self._renderer.preview_color_limits(
                        candidate.value.low,
                        candidate.value.high,
                    )
                    self._render_current(RenderEffect.OVERLAY, schedule_fit=False)
            return
        if self._selector_controller.active_gesture is None:
            self._cancel_gesture()
            return
        current = self._selector_controller.candidate_state()
        if current is None:
            self._cancel_gesture()
            return
        updated = self._selector_controller.pointer_move(
            point.x,
            point.y,
            x_bounds=self._selector_x_bounds(gesture.transform),
            y_bounds=self._selector_y_bounds(gesture.transform),
        )
        if updated is not None and updated != current:
            assert self._renderer is not None
            displayed = (
                self._projected._display_selector_state(updated)
                if self._view is not None
                else self._special_display_selector_state(updated)
            )
            if not gesture.external_scene:
                with self._renderer.raster_transaction():
                    self._renderer.preview_selector(displayed)
            self._emit_selection(SelectionChange.UPDATED, updated)

    def _on_button_release(self, event: Any) -> None:
        gesture = self._gesture
        if gesture is None:
            return
        if getattr(event, "button", None) == 2:
            if not isinstance(gesture, _PanGesture):
                return
            try:
                self._update_pan(event, gesture, render=False)
            finally:
                self._clear_gesture(gesture)
            if gesture.candidate is not None:
                self._set_viewport_state(gesture.candidate)
            return
        if isinstance(gesture, _PanGesture):
            return
        if isinstance(gesture, _ColorGesture):
            self._finish_color_gesture(event, gesture)
            return
        self._finish_selector_gesture(event, gesture)

    def _finish_color_gesture(
        self,
        event: Any,
        gesture: _ColorGesture,
    ) -> None:
        point = self._event_canonical(
            event,
            captured_transform=gesture.transform,
        )
        if point is None:
            self._cancel_gesture()
            return
        gesture.drag = gesture.drag.moved(point.x)
        candidate = self._display_color_limit_candidate()
        assert candidate is not None
        self._clear_gesture(gesture)
        try:
            if gesture.drag.changed:
                self.set_color_limits(
                    candidate.value.low,
                    candidate.value.high,
                    fixed=True,
                )
            else:
                self._render_current(
                    RenderEffect.BASE_GEOMETRY,
                    schedule_fit=False,
                )
        except BaseException:
            self._render_current(
                RenderEffect.BASE_GEOMETRY,
                schedule_fit=False,
            )
            raise

    def _finish_selector_gesture(
        self,
        event: Any,
        gesture: _SelectorGesture,
    ) -> None:
        if self._selector_controller.active_gesture is None:
            self._cancel_gesture()
            return
        try:
            point = self._event_canonical(
                event,
                captured_transform=gesture.transform,
            )
            if point is None:
                self._cancel_gesture()
                return
            candidate = self._selector_controller.finish_gesture(
                point.x,
                point.y,
                x_bounds=self._selector_x_bounds(gesture.transform),
                y_bounds=self._selector_y_bounds(gesture.transform),
            )
        except BaseException:
            self._cancel_gesture()
            raise
        self._clear_gesture(gesture)
        if candidate is None:
            return
        if not self._is_degenerate_selector(candidate):
            stored = self._install_selector_state(
                candidate,
                emit_change=False,
                finished_gesture=gesture,
            )
            self._emit_selection(SelectionChange.COMMITTED, stored)
            return
        committed = self._projected._selector_state_or_none(candidate.kind)
        if committed is not None and committed.kind is SelectorKind.AREA:
            self.remove_selector(committed.kind)
            return
        self._render_current(RenderEffect.OVERLAY)

    def _on_figure_leave(self, _event: Any) -> None:
        self._cancel_gesture()

    def _on_key_press(self, event: Any) -> None:
        if str(getattr(event, "key", "")).lower() != "escape":
            return
        self._cancel_gesture()
        if (
            isinstance(self._spec, FacetGridPlot)
            and self._renderer is not None
            and self._facet_focus_index is not None
        ):
            self.show_facet_overview()

    @staticmethod
    def _is_degenerate_selector(state: SelectorState) -> bool:
        if state.kind is SelectorKind.X_RANGE:
            assert isinstance(state.value, NumericRange)
            return state.value.span <= 0.0
        if state.kind is SelectorKind.AREA:
            assert isinstance(state.value, RectangleRange)
            return state.value.x.span <= 0.0 or state.value.y.span <= 0.0
        return False

    @classmethod
    def _require_stable_selector(cls, state: SelectorState) -> None:
        if not cls._is_degenerate_selector(state):
            return
        if state.kind is SelectorKind.AREA:
            raise ValueError("area selector requires non-degenerate x and y ranges")
        raise ValueError(f"{state.kind.value} selector requires a non-degenerate range")

    def _update_pan(
        self,
        event: Any,
        gesture: _PanGesture,
        *,
        render: bool = True,
    ) -> bool:
        assert self._renderer is not None
        point = gesture.transform.display_point(event, self._renderer.figure.canvas)
        if point is None:
            return False
        dx = gesture.origin.x - point.x
        image_like = isinstance(self._projected._semantic_spec(), ImagePlot)
        dy = gesture.origin.y - point.y
        if math.isclose(dx, 0.0, rel_tol=0.0, abs_tol=1.0e-15) and (
            not image_like
            or math.isclose(dy, 0.0, rel_tol=0.0, abs_tol=1.0e-15)
        ):
            return False
        moved_x = gesture.x.shifted(dx)
        moved_y = gesture.y.shifted(dy) if image_like else gesture.y
        selected_x = self._viewport_x_from_axes(moved_x)
        selected = RectangleRange(selected_x, moved_y)
        current = gesture.candidate or self._projected.viewport
        if current is not None and np.allclose(
            (current.x.low, current.x.high, current.y.low, current.y.high),
            (selected.x.low, selected.x.high, selected.y.low, selected.y.high),
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            return False
        gesture.candidate = selected
        if render:
            self._render_current(
                RenderEffect.AXIS_TRANSFORM | RenderEffect.OVERLAY,
                schedule_fit=False,
            )
        return True

    def _clear_gesture(self, gesture: _PointerGesture) -> None:
        if self._gesture is not gesture:
            return
        self._gesture = None
        if isinstance(gesture, _PanGesture) or self._renderer is None:
            return
        try:
            self._renderer.set_selector_scene_exclusion(None)
        finally:
            self._renderer.end_selector_gesture()

    def _cancel_gesture(self) -> SelectorState | None:
        gesture = self._gesture
        if gesture is None:
            return None
        cancelled = None
        try:
            if isinstance(gesture, _SelectorGesture):
                cancelled = self._selector_controller.lost_pointer_capture()
        finally:
            self._clear_gesture(gesture)
        if (
            isinstance(gesture, _PanGesture)
            and gesture.candidate is not None
            and self._renderer is not None
            and not self._closed
        ):
            self._render_current(
                RenderEffect.AXIS_TRANSFORM | RenderEffect.OVERLAY,
                schedule_fit=False,
            )
        restore_committed = isinstance(gesture, _ColorGesture) or cancelled is not None
        if restore_committed and self._renderer is not None and not self._closed:
            color_gesture = isinstance(gesture, _ColorGesture)
            self._render_current(
                RenderEffect.BASE_GEOMETRY if color_gesture else RenderEffect.OVERLAY,
                schedule_fit=False,
            )
        return cancelled

    def save(
        self,
        path: str | Path,
        *,
        dpi: float | None = None,
        export_scale: float | None = None,
        **kwargs: Any,
    ) -> None:
        if dpi is not None and export_scale is not None:
            raise ValueError("specify dpi or export_scale, not both")
        if isinstance(dpi, bool) or isinstance(export_scale, bool):
            raise TypeError("dpi and export_scale must be positive numbers or None")
        selected_dpi = self._defaults.layout.export_dpi if dpi is None else float(dpi)
        if export_scale is not None:
            selected_dpi = self.surface_plan.logical_dpi * float(export_scale)
        if not math.isfinite(selected_dpi) or selected_dpi <= 0.0:
            raise ValueError("export dpi must be a positive finite number")
        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                self._renderer.save(path, dpi=selected_dpi, **kwargs)

    def rgba(self) -> np.ndarray:
        with self._render_lock:
            with self._lock:
                self._assert_open()
            assert self._renderer is not None
            with self._renderer.raster_transaction():
                return self._renderer.rgba()

    def close(self) -> None:
        logical_completion: Future[FitResult] | None = None
        with self._render_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._cancel_gesture()
                self._fit_cancel.set()
                self._fit_context_generation += 1
                self._fit_request_generation += 1
                logical_completion = self._live_fit_completion
                self._live_fit_completion = None
                self._live_fit_request = None
                self._live_fit_pending = False
                self._surface_callbacks.clear()
                self._display_callbacks.clear()
                self._fit_callbacks.clear()
                self._selection_subscriptions.clear()
        if logical_completion is not None and not logical_completion.done():
            logical_completion.set_exception(RuntimeError("plot session is closed"))
        caller_name = current_thread().name
        self._fit_executor.shutdown(
            wait=not caller_name.startswith(f"{_FIT_THREAD_PREFIX}_"),
            cancel_futures=True,
        )

    def __enter__(self) -> "PlotSession":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


__all__ = [
    "DisplayDescription",
    "FitEvent",
    "FitScope",
    "FitSelection",
    "PlotInput",
    "PlotSession",
    "PlotSessionConfig",
    "PulseTimelineSelectionData",
    "SelectionChange",
    "SelectionData",
    "SelectionEvent",
    "SelectorData",
    "SessionRevisions",
]
