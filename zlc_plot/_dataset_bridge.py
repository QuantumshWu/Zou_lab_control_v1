"""Private, read-only bridge from :mod:`zlc_data` into the plot engine.

The authoritative dataset contract stays in ``zlc_data``.  These small views
only expose the names used by the accepted plotting core; they never own data,
schema identity, provenance, or a signal lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, RLock
from types import MappingProxyType
from typing import Any, Generic, TypeVar

import numpy as np

from zlc_data import DatasetSchema, OwnedSnapshot
from zlc_data.axis import AxisSpec
from zlc_data.schema import GridTopology, PointColumn, PointTable
from zlc_data.value import expand_dataset_validity


class RevisionError(ValueError):
    """A plot-side revision moved backwards or was reused for new content."""


@dataclass(frozen=True, slots=True)
class _Unit:
    symbol: str
    dimension: str
    scale: float = 1.0
    offset: float = 0.0

    def compatible_with(self, other: "_Unit") -> bool:
        return isinstance(other, _Unit) and self.dimension == other.dimension

    def to_canonical(self, values: Any) -> np.ndarray:
        array = np.asarray(values)
        if self.scale == 1.0 and self.offset == 0.0:
            return array
        return array * self.scale + self.offset

    def from_canonical(self, values: Any) -> np.ndarray:
        array = np.asarray(values)
        if self.scale == 1.0 and self.offset == 0.0:
            return array
        return (array - self.offset) / self.scale

    def convert_value_to(self, values: Any, target: "_Unit") -> np.ndarray:
        if not self.compatible_with(target):
            raise ValueError(
                f"incompatible display units {self.symbol!r} and {target.symbol!r}"
            )
        array = np.asarray(values)
        if self == target:
            return array
        return target.from_canonical(self.to_canonical(array))


_KNOWN_UNITS = (
    _Unit("1", "dimensionless"),
    _Unit("m", "length"),
    _Unit("km", "length", 1e3),
    _Unit("cm", "length", 1e-2),
    _Unit("mm", "length", 1e-3),
    _Unit("um", "length", 1e-6),
    _Unit("nm", "length", 1e-9),
    _Unit("s", "time"),
    _Unit("ms", "time", 1e-3),
    _Unit("us", "time", 1e-6),
    _Unit("ns", "time", 1e-9),
    _Unit("min", "time", 60.0),
    _Unit("h", "time", 3600.0),
    _Unit("Hz", "frequency"),
    _Unit("kHz", "frequency", 1e3),
    _Unit("MHz", "frequency", 1e6),
    _Unit("GHz", "frequency", 1e9),
    _Unit("V", "voltage"),
    _Unit("mV", "voltage", 1e-3),
    _Unit("uV", "voltage", 1e-6),
    _Unit("A", "current"),
    _Unit("mA", "current", 1e-3),
    _Unit("uA", "current", 1e-6),
    _Unit("K", "temperature"),
    _Unit("degC", "temperature", 1.0, 273.15),
    _Unit("rad", "angle"),
    _Unit("deg", "angle", np.pi / 180.0),
    _Unit("count", "count"),
)

_UNIT_ALIASES = {
    "arb": "1",
    "µm": "um",
    "μm": "um",
    "µs": "us",
    "μs": "us",
    "µV": "uV",
    "μV": "uV",
    "µA": "uA",
    "μA": "uA",
    "°C": "degC",
    "°": "deg",
}


class _UnitTable:
    """Private display conversion table over producer-owned unit strings."""

    def __init__(self) -> None:
        self._known = {unit.symbol: unit for unit in _KNOWN_UNITS}
        self._opaque: dict[str, _Unit] = {}

    def resolve(self, symbol: str | _Unit | None) -> _Unit:
        if isinstance(symbol, _Unit):
            return symbol
        selected = "1" if symbol is None else str(symbol).strip()
        if not selected:
            selected = "1"
        selected = _UNIT_ALIASES.get(selected, selected)
        known = self._known.get(selected)
        if known is not None:
            return known
        unit = self._opaque.get(selected)
        if unit is None:
            unit = _Unit(selected, f"opaque:{selected}")
            self._opaque[selected] = unit
        return unit

    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted((*self._known, *self._opaque)))


_UNITS = _UnitTable()


def _resolve_unit(
    value: str | _Unit | None,
    registry: _UnitTable | None = None,
) -> _Unit:
    return (registry or _UNITS).resolve(value)


def _readonly(array: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.asarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def _coordinates(axis: AxisSpec) -> np.ndarray:
    if axis.coordinates is None:
        return _readonly(np.arange(axis.index_origin, axis.index_origin + axis.size))
    return _readonly(axis.coordinates)


@dataclass(frozen=True, slots=True)
class _AxisView:
    name: str
    size: int
    coordinates: np.ndarray
    label: str
    role: str
    canonical_unit: _Unit
    display_unit: _Unit


def _axis_view(axis: AxisSpec) -> _AxisView:
    unit = _resolve_unit(axis.unit)
    return _AxisView(
        name=axis.axis_id.value,
        size=axis.size,
        coordinates=_coordinates(axis),
        label=axis.name,
        role=axis.role.value,
        canonical_unit=unit,
        display_unit=unit,
    )


def _point_values(column: PointColumn) -> np.ndarray:
    if column.value_kind == PointColumn.NUMERIC and any(
        value is None for value in column.values
    ):
        return _readonly(
            tuple(np.nan if value is None else value for value in column.values),
            dtype=float,
        )
    return _readonly(column.values)


def _point_axis_view(column: PointColumn) -> _AxisView:
    unit = _resolve_unit(column.unit)
    return _AxisView(
        name=column.coordinate_id.value,
        size=len(column.values),
        coordinates=_point_values(column),
        label=column.name,
        role=column.role.value,
        canonical_unit=unit,
        display_unit=unit,
    )


@dataclass(frozen=True, slots=True)
class _PointTableView:
    source: PointTable
    columns: tuple[_AxisView, ...]

    @property
    def row_count(self) -> int:
        return self.source.row_count

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def axis(self, name: str) -> _AxisView:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class _PointTopologyView:
    source: GridTopology
    dimensions: tuple[_AxisView, ...]
    row_to_cell: np.ndarray

    def dimension(self, name: str) -> _AxisView:
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class _SchemaView:
    source: DatasetSchema
    generation: str
    repeat_axis: _AxisView
    point_table: _PointTableView
    data_axes: tuple[_AxisView, ...]
    point_topology: _PointTopologyView | None
    value_label: str
    canonical_unit: _Unit
    display_unit: _Unit
    dtype: np.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source.physical_shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def R(self) -> int:
        return self.source.repeat_axis.size

    @property
    def P(self) -> int:
        return self.source.point_table.row_count

    def exactly_equals(self, other: object) -> bool:
        return (
            isinstance(other, _SchemaView)
            and self.generation == other.generation
            and self.source == other.source
        )

    def assert_exact(self, other: object) -> None:
        """Require the producer generation and immutable schema value.

        Schema equality is delegated to the authoritative zlc_data value.  A
        producer may reconstruct an equal immutable schema for each snapshot;
        object identity is not a data contract.  A new generation or any
        semantic schema change still requires a new plot session.
        """

        if not self.exactly_equals(other):
            raise ValueError("plot update changed the authoritative DatasetSchema")


def _schema_view(snapshot: OwnedSnapshot) -> _SchemaView:
    source = snapshot.block.schema
    point_table = _PointTableView(
        source.point_table,
        tuple(_point_axis_view(column) for column in source.point_table.columns),
    )
    topology = source.grid_topology
    topology_view = None
    if topology is not None:
        by_id = {
            column.coordinate_id: column for column in source.point_table.columns
        }
        dimensions = []
        for axis_id, domain in zip(
            topology.dimension_ids,
            topology.coordinate_domains,
            strict=True,
        ):
            column = by_id[axis_id]
            unit = _resolve_unit(column.unit)
            dimensions.append(
                _AxisView(
                    name=axis_id.value,
                    size=len(domain),
                    coordinates=_readonly(domain),
                    label=column.name,
                    role=column.role.value,
                    canonical_unit=unit,
                    display_unit=unit,
                )
            )
        topology_view = _PointTopologyView(
            topology,
            tuple(dimensions),
            _readonly(topology.row_to_cell, dtype=np.int64),
        )
    value_unit = _resolve_unit(source.cell_schema.value_unit)
    return _SchemaView(
        source=source,
        generation=snapshot.ref.stream_generation.value,
        repeat_axis=_axis_view(source.repeat_axis),
        point_table=point_table,
        data_axes=tuple(_axis_view(axis) for axis in source.cell_schema.data_axes),
        point_topology=topology_view,
        value_label="value",
        canonical_unit=value_unit,
        display_unit=value_unit,
        dtype=source.cell_schema.dtype,
    )


@dataclass(frozen=True, slots=True, eq=False)
class _ReadonlyDataset:
    """Zero-copy plotting view over one authoritative ``OwnedSnapshot``."""

    source: OwnedSnapshot
    schema: _SchemaView
    values: np.ndarray
    validity: np.ndarray
    revision: int
    metadata: MappingProxyType

    @property
    def shape(self) -> tuple[int, ...]:
        return self.schema.shape

    @property
    def generation(self) -> str:
        return self.schema.generation

    def displayed_values(self) -> np.ndarray:
        return self.schema.canonical_unit.convert_value_to(
            self.values,
            self.schema.display_unit,
        )


def bridge_snapshot(snapshot: OwnedSnapshot | _ReadonlyDataset) -> _ReadonlyDataset:
    if isinstance(snapshot, _ReadonlyDataset):
        return snapshot
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("plot data must be zlc_data.OwnedSnapshot")
    values = snapshot.block.values
    validity = expand_dataset_validity(snapshot.block.validity, snapshot.block.schema)
    values.setflags(write=False)
    validity.setflags(write=False)
    return _ReadonlyDataset(
        source=snapshot,
        schema=_schema_view(snapshot),
        values=values,
        validity=validity,
        revision=snapshot.ref.revision.value,
        metadata=MappingProxyType({}),
    )


ItemT = TypeVar("ItemT")


@dataclass(frozen=True, slots=True)
class _RevisionedItem(Generic[ItemT]):
    revision: int
    payload: ItemT


@dataclass(frozen=True, slots=True)
class _IngressMetrics:
    published: int
    consumed: int
    coalesced_updates: int
    pending: bool
    last_published_revision: int | None
    last_consumed_revision: int | None


class _LatestRevisionSlot(Generic[ItemT]):
    """Private capacity-one handoff for monitor display updates."""

    def __init__(self, *, initial_revision: int | None = None) -> None:
        self._condition = Condition(RLock())
        self._floor = initial_revision
        self._last = initial_revision
        self._last_consumed: int | None = None
        self._pending: _RevisionedItem[ItemT] | None = None
        self._published = 0
        self._consumed = 0
        self._coalesced = 0
        self._closed = False

    def publish(self, revision: int, payload: ItemT) -> _RevisionedItem[ItemT]:
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise RevisionError("revision must be a non-negative integer")
        with self._condition:
            if self._closed:
                raise RuntimeError("latest plot slot is closed")
            if self._last is not None and revision <= self._last:
                raise RevisionError(
                    f"revision {revision} is not newer than {self._last}"
                )
            if self._pending is not None:
                self._coalesced += 1
            item = _RevisionedItem(revision, payload)
            self._pending = item
            self._last = revision
            self._published += 1
            self._condition.notify_all()
            return item

    def take_latest(self) -> _RevisionedItem[ItemT] | None:
        with self._condition:
            item = self._pending
            if item is not None:
                self._pending = None
                self._consumed += 1
                self._last_consumed = item.revision
            return item

    def wait_latest(self, timeout: float | None = None) -> _RevisionedItem[ItemT] | None:
        with self._condition:
            if self._pending is None and not self._closed:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._closed,
                    timeout=timeout,
                )
            item = self._pending
            if item is not None:
                self._pending = None
                self._consumed += 1
                self._last_consumed = item.revision
            return item

    def metrics(self) -> _IngressMetrics:
        return _IngressMetrics(
            published=self._published,
            consumed=self._consumed,
            coalesced_updates=self._coalesced,
            pending=self._pending is not None,
            last_published_revision=self._last,
            last_consumed_revision=self._last_consumed,
        )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()


__all__: tuple[str, ...] = ()
