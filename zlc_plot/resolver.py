"""Schema-driven plot defaults and physical-axis coverage.

This is the single place that maps authoritative dataset axes into plot-side
``AxisRef`` values.  It never examines array rank, singleton shape patterns,
axis names, or current data values to guess a physical role.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from zlc_data import DatasetSchema
from zlc_data.axis import SCALAR, SPATIAL_X, SPATIAL_Y

from ._dataset_bridge import _SchemaView
from .kinds import AxisDomain, AxisRef, PlotKind
from .parameters import ParameterSchema, ParameterSpec, RenderEffect
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotLabels,
    PlotSpec,
    PulseTimelinePlot,
    Reduction,
    RollingPlot,
)


@dataclass(frozen=True, slots=True)
class _AxisRecord:
    ref: AxisRef
    size: int
    label: str
    role: str


_INDEX_EFFECTS = (
    RenderEffect.VIEW_PROJECTION
    | RenderEffect.PAYLOAD_PROJECTION
    | RenderEffect.BASE_GEOMETRY
    | RenderEffect.AXIS_TRANSFORM
    | RenderEffect.TEXT
    | RenderEffect.CHROME
    | RenderEffect.OVERLAY
    | RenderEffect.FIT_SELECTION
    | RenderEffect.INTERACTION_REPROJECT
)


def _records(schema: DatasetSchema | _SchemaView) -> tuple[_AxisRecord, ...]:
    if isinstance(schema, DatasetSchema):
        repeat = schema.repeat_axis
        point_columns = schema.point_table.columns
        topology = schema.grid_topology
        data_axes = schema.cell_schema.data_axes
        result = [
            _AxisRecord(
                AxisRef.repeat(), repeat.size, repeat.name, repeat.role.value
            ),
            _AxisRecord(
                AxisRef.point_rows(),
                schema.point_table.row_count,
                "Point row",
                "scan-point",
            ),
        ]
        result.extend(
            _AxisRecord(
                AxisRef.point(column.coordinate_id.value),
                schema.point_table.row_count,
                column.name,
                column.role.value,
            )
            for column in point_columns
        )
        if topology is not None:
            by_id = {column.coordinate_id: column for column in point_columns}
            result.extend(
                _AxisRecord(
                    AxisRef.point_dimension(axis_id.value),
                    len(domain),
                    by_id[axis_id].name,
                    by_id[axis_id].role.value,
                )
                for axis_id, domain in zip(
                    topology.dimension_ids,
                    topology.coordinate_domains,
                    strict=True,
                )
            )
        result.extend(
            _AxisRecord(
                AxisRef.data(axis.axis_id.value),
                axis.size,
                axis.name,
                axis.role.value,
            )
            for axis in data_axes
        )
        return tuple(result)

    if not isinstance(schema, _SchemaView):
        raise TypeError("schema must be zlc_data.DatasetSchema")
    result = [
        _AxisRecord(
            AxisRef.repeat(),
            schema.repeat_axis.size,
            schema.repeat_axis.label,
            schema.repeat_axis.role,
        ),
        _AxisRecord(
            AxisRef.point_rows(), schema.P, "Point row", "scan-point"
        ),
    ]
    result.extend(
        _AxisRecord(
            AxisRef.point(column.name), column.size, column.label, column.role
        )
        for column in schema.point_table.columns
    )
    if schema.point_topology is not None:
        result.extend(
            _AxisRecord(
                AxisRef.point_dimension(axis.name), axis.size, axis.label, axis.role
            )
            for axis in schema.point_topology.dimensions
        )
    result.extend(
        _AxisRecord(AxisRef.data(axis.name), axis.size, axis.label, axis.role)
        for axis in schema.data_axes
    )
    return tuple(result)


def axis_choices(
    schema: DatasetSchema,
    *,
    domain: AxisDomain | None = None,
) -> tuple[AxisRef, ...]:
    """Return declared axes in stable producer order, optionally by domain."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if domain is not None and not isinstance(domain, AxisDomain):
        raise TypeError("domain must be AxisDomain or None")
    return tuple(
        record.ref
        for record in _records(schema)
        if domain is None or record.ref.domain is domain
    )


def default_plot_spec(schema: DatasetSchema, kind: PlotKind) -> PlotSpec:
    """Derive a non-destructive initial plot from declared axis roles only."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(kind, PlotKind):
        raise TypeError("kind must be PlotKind")
    records = _records(schema)
    if kind is PlotKind.PULSE_TIMELINE:
        return PulseTimelinePlot()
    if kind is PlotKind.HISTOGRAM:
        # Repeat is the only physical domain whose declared acquisition
        # meaning makes pooling a safe initial presentation.  Point rows and
        # data axes stay visibly indexed until the user explicitly enables
        # their Histogram sample controls.
        samples = (
            (AxisRef.repeat(),)
            if schema.repeat_axis.size > 1
            else ()
        )
        return HistogramPlot(samples=samples)
    if kind is PlotKind.ROLLING:
        return RollingPlot()
    if kind is PlotKind.FACET_GRID:
        raise ValueError("FacetGrid requires an explicit facet and cell PlotSpec")

    data_records = tuple(
        record
        for record in records
        if record.ref.domain is AxisDomain.DATA and record.role != SCALAR.value
    )
    point_columns = tuple(
        record
        for record in records
        if record.ref.domain is AxisDomain.POINT_COORDINATE
    )
    if kind is PlotKind.IMAGE:
        xs = tuple(record for record in data_records if record.role == SPATIAL_X.value)
        ys = tuple(record for record in data_records if record.role == SPATIAL_Y.value)
        if len(xs) != 1 or len(ys) != 1:
            raise ValueError(
                "Image default requires exactly one declared spatial-x and spatial-y axis"
            )
        return ImagePlot(xs[0].ref, ys[0].ref)

    if kind is not PlotKind.CURVE:
        raise ValueError(f"unsupported default plot kind {kind.value!r}")
    candidates = tuple(record for record in point_columns if record.size > 1)
    if len(candidates) == 1:
        return CurvePlot(candidates[0].ref)
    if len(candidates) > 1:
        raise ValueError("Curve default has multiple equal-priority point coordinates")
    nontrivial_data = tuple(record for record in data_records if record.size > 1)
    if schema.point_table.row_count > 1:
        return CurvePlot(AxisRef.point_rows())
    if len(nontrivial_data) == 1:
        return CurvePlot(nontrivial_data[0].ref)
    raise ValueError("Curve default needs an explicit x axis")


def _semantic_refs(spec: PlotSpec) -> tuple[AxisRef, ...]:
    semantic: Any = spec.cell if isinstance(spec, FacetGridPlot) else spec
    refs: list[AxisRef] = []
    for name in ("x", "y", "group"):
        value = getattr(semantic, name, None)
        if isinstance(value, AxisRef):
            refs.append(value)
    if isinstance(semantic, HistogramPlot):
        refs.extend(semantic.samples)
    if isinstance(spec, FacetGridPlot):
        refs.append(spec.facet)
    return tuple(refs)


def _index_parameter_name(ref: AxisRef) -> str:
    suffix = "" if ref.axis_id is None else f"__{ref.axis_id}"
    return f"index__{ref.domain.value}{suffix}"


def _point_coverage(refs: tuple[AxisRef, ...]) -> tuple[bool, frozenset[str]]:
    row_domain = any(
        ref.domain in {AxisDomain.POINT_ROW, AxisDomain.POINT_COORDINATE}
        for ref in refs
    )
    dimensions = frozenset(
        ref.axis_id
        for ref in refs
        if ref.domain is AxisDomain.POINT_DIMENSION and ref.axis_id is not None
    )
    if row_domain and dimensions:
        raise ValueError(
            "point rows/coordinates and GridTopology dimensions cannot cover P twice"
        )
    return row_domain, dimensions


def index_parameter_specs(
    schema: DatasetSchema | _SchemaView,
    spec: PlotSpec,
) -> tuple[tuple[ParameterSpec[object], ...], dict[str, AxisRef]]:
    """Return visible index controls for every otherwise-uncovered physical axis."""

    refs = _semantic_refs(spec)
    if len(set(refs)) != len(refs):
        raise ValueError("a physical axis cannot fill multiple plot roles")
    record_by_ref = {record.ref: record for record in _records(schema)}
    missing = tuple(ref for ref in refs if ref not in record_by_ref)
    if missing:
        raise ValueError(f"PlotSpec refers to undeclared axis {missing[0]!r}")
    row_covered, dimensions = _point_coverage(refs)
    topology_dimensions = tuple(
        record
        for record in record_by_ref.values()
        if record.ref.domain is AxisDomain.POINT_DIMENSION
    )
    indexed: list[_AxisRecord] = []
    semantic = spec.cell if isinstance(spec, FacetGridPlot) else spec
    if isinstance(semantic, ImagePlot):
        point_domains = {
            AxisDomain.POINT_ROW,
            AxisDomain.POINT_COORDINATE,
            AxisDomain.POINT_DIMENSION,
        }
        if semantic.x.domain in point_domains and semantic.y.domain in point_domains:
            if (
                semantic.x.domain is not AxisDomain.POINT_DIMENSION
                or semantic.y.domain is not AxisDomain.POINT_DIMENSION
                or (
                    schema.grid_topology
                    if isinstance(schema, DatasetSchema)
                    else schema.point_topology
                ) is None
            ):
                raise ValueError(
                    "two physical point axes require distinct declared "
                    "GridTopology dimensions"
                )

    repeat = record_by_ref[AxisRef.repeat()]
    repeat_explicit = AxisRef.repeat() in refs
    repeat_reduced = isinstance(semantic, (CurvePlot, ImagePlot, RollingPlot))
    if not repeat_explicit and not repeat_reduced and repeat.size > 1:
        indexed.append(repeat)

    if dimensions:
        indexed.extend(
            record
            for record in topology_dimensions
            if record.ref.axis_id not in dimensions and record.size > 1
        )
    elif not row_covered and (
        schema.point_table.row_count
        if isinstance(schema, DatasetSchema)
        else schema.P
    ) > 1:
        indexed.append(record_by_ref[AxisRef.point_rows()])

    for record in record_by_ref.values():
        if record.ref.domain is not AxisDomain.DATA or record.ref in refs:
            continue
        if record.role == SCALAR.value:
            continue
        if record.size > 1:
            indexed.append(record)

    mapping: dict[str, AxisRef] = {}
    entries: list[ParameterSpec[object]] = []
    for record in indexed:
        name = _index_parameter_name(record.ref)
        mapping[name] = record.ref
        entries.append(
            ParameterSpec(
                name,
                int,
                _INDEX_EFFECTS,
                default=0,
                label=f"{record.label} index",
                choices=tuple(range(record.size)),
                minimum=0,
                maximum=record.size - 1,
                step=1,
            )
        )
    return tuple(entries), mapping


def _spec_sample_name(ref: AxisRef) -> str:
    suffix = "axis" if ref.axis_id is None else ref.axis_id
    return f"sample__{ref.domain.value}__{suffix}"


def _semantic_authoring_axes(schema: DatasetSchema) -> tuple[AxisRef, ...]:
    return tuple(
        record.ref
        for record in _records(schema)
        if not (
            record.ref.domain is AxisDomain.DATA
            and record.role == SCALAR.value
        )
    )


def _plot_spec_authoring_schema(
    schema: DatasetSchema,
    kind: PlotKind,
) -> tuple[ParameterSchema, dict[str, AxisRef]]:
    """Build the one headless PlotSpec draft schema used by every frontend."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(kind, PlotKind):
        raise TypeError("kind must be PlotKind")
    axes = _semantic_authoring_axes(schema)
    effects = _INDEX_EFFECTS
    entries: list[ParameterSpec[object]] = []
    if kind is PlotKind.FACET_GRID:
        entries.extend((
            ParameterSpec(
                "facet",
                AxisRef,
                effects,
                default=None,
                allow_none=True,
                choices=axes,
                label="Facet",
            ),
            ParameterSpec(
                "cell_kind",
                str,
                effects,
                default="curve",
                choices=("curve", "image", "histogram"),
                label="Cell",
            ),
        ))
    semantic_kind = PlotKind.CURVE if kind is PlotKind.FACET_GRID else kind
    if semantic_kind in {PlotKind.CURVE, PlotKind.IMAGE} or kind is PlotKind.FACET_GRID:
        entries.append(ParameterSpec(
            "x",
            AxisRef,
            effects,
            default=None,
            allow_none=True,
            choices=axes,
            label="X axis",
        ))
    if semantic_kind is PlotKind.IMAGE or kind is PlotKind.FACET_GRID:
        entries.append(ParameterSpec(
            "y",
            AxisRef,
            effects,
            default=None,
            allow_none=True,
            choices=axes,
            label="Y axis",
        ))
    if semantic_kind in {PlotKind.CURVE, PlotKind.ROLLING} or kind is PlotKind.FACET_GRID:
        entries.append(ParameterSpec(
            "group",
            AxisRef,
            effects,
            default=None,
            allow_none=True,
            choices=axes,
            label="Group",
        ))
    if semantic_kind in {PlotKind.CURVE, PlotKind.IMAGE, PlotKind.ROLLING} or kind is PlotKind.FACET_GRID:
        entries.append(ParameterSpec(
            "reduction",
            Reduction,
            effects,
            default=Reduction.MEAN,
            choices=tuple(Reduction),
            label="Reduction",
        ))
    sample_refs: dict[str, AxisRef] = {}
    if semantic_kind is PlotKind.HISTOGRAM or kind is PlotKind.FACET_GRID:
        for ref in axes:
            name = _spec_sample_name(ref)
            sample_refs[name] = ref
            entries.append(ParameterSpec(
                name,
                bool,
                effects,
                default=False,
                label=f"Sample · {ref}",
            ))
    return ParameterSchema(entries), sample_refs


def _semantic_spec(spec: PlotSpec) -> PlotSpec:
    return spec.cell if isinstance(spec, FacetGridPlot) else spec


def _plot_spec_authoring_values(
    schema: DatasetSchema,
    kind: PlotKind,
    spec: PlotSpec | None,
) -> dict[str, object]:
    parameter_schema, sample_refs = _plot_spec_authoring_schema(schema, kind)
    values = dict(parameter_schema.initial_values())
    if spec is None:
        return values
    if spec.kind is not kind:
        raise ValueError("PlotSpec kind differs from its authoring kind")
    semantic = _semantic_spec(spec)
    if isinstance(spec, FacetGridPlot):
        values["facet"] = spec.facet
        values["cell_kind"] = semantic.kind.value
    if isinstance(semantic, (CurvePlot, ImagePlot)):
        values["x"] = semantic.x
    if isinstance(semantic, ImagePlot):
        values["y"] = semantic.y
    if isinstance(semantic, (CurvePlot, RollingPlot)):
        values["group"] = semantic.group
    if isinstance(semantic, (CurvePlot, ImagePlot, RollingPlot)):
        values["reduction"] = semantic.reduction
    selected_samples = (
        set(semantic.samples) if isinstance(semantic, HistogramPlot) else set()
    )
    for name, ref in sample_refs.items():
        values[name] = ref in selected_samples
    return values


def _plot_spec_authoring_control_names(
    schema: DatasetSchema,
    kind: PlotKind,
    values: dict[str, object],
) -> tuple[str, ...]:
    _parameter_schema, sample_refs = _plot_spec_authoring_schema(schema, kind)
    names: list[str] = []
    semantic_kind = kind
    if kind is PlotKind.FACET_GRID:
        names.extend(("facet", "cell_kind"))
        selected = values.get("cell_kind")
        semantic_kind = PlotKind(selected) if selected is not None else PlotKind.CURVE
    if semantic_kind is PlotKind.CURVE:
        names.extend(("x", "group", "reduction"))
    elif semantic_kind is PlotKind.IMAGE:
        names.extend(("x", "y", "reduction"))
    elif semantic_kind is PlotKind.HISTOGRAM:
        names.extend(sample_refs)
    elif semantic_kind is PlotKind.ROLLING:
        names.extend(("group", "reduction"))
    return tuple(names)


def resolve_plot_spec(
    schema: DatasetSchema,
    kind: PlotKind,
    values: Mapping[str, object],
    *,
    labels: PlotLabels = PlotLabels(),
) -> PlotSpec:
    """Resolve a headless authoring mapping into one validated PlotSpec."""

    if not isinstance(values, Mapping):
        raise TypeError("plot specification values must be a mapping")
    if not isinstance(labels, PlotLabels):
        raise TypeError("labels must be PlotLabels")
    parameter_schema, sample_refs = _plot_spec_authoring_schema(schema, kind)
    prepared = parameter_schema.initial_values(values)
    semantic_kind = kind
    if kind is PlotKind.FACET_GRID:
        semantic_kind = PlotKind(prepared["cell_kind"])

    def required(name: str) -> AxisRef:
        value = prepared[name]
        if not isinstance(value, AxisRef):
            raise ValueError(f"PlotSpec needs an explicit {name.replace('_', ' ')}")
        return value

    if semantic_kind is PlotKind.CURVE:
        semantic: PlotSpec = CurvePlot(
            required("x"),
            group=prepared["group"],
            reduction=prepared["reduction"],
            labels=labels,
        )
    elif semantic_kind is PlotKind.IMAGE:
        semantic = ImagePlot(
            required("x"),
            required("y"),
            reduction=prepared["reduction"],
            labels=labels,
        )
    elif semantic_kind is PlotKind.HISTOGRAM:
        semantic = HistogramPlot(
            samples=tuple(
                ref for name, ref in sample_refs.items() if bool(prepared[name])
            ),
            labels=labels,
        )
    elif semantic_kind is PlotKind.ROLLING:
        semantic = RollingPlot(
            group=prepared["group"],
            reduction=prepared["reduction"],
            labels=labels,
        )
    elif semantic_kind is PlotKind.PULSE_TIMELINE:
        semantic = PulseTimelinePlot(labels=labels)
    else:
        raise ValueError(f"unsupported authored plot kind {semantic_kind.value!r}")
    spec = (
        FacetGridPlot(required("facet"), semantic, labels=labels)
        if kind is PlotKind.FACET_GRID
        else semantic
    )
    index_parameter_specs(schema, spec)
    return spec


def indices_from_parameters(
    mapping: dict[str, AxisRef],
    values: Any,
) -> dict[AxisRef, int]:
    return {ref: int(values[name]) for name, ref in mapping.items()}


__all__ = ["axis_choices", "default_plot_spec", "resolve_plot_spec"]
