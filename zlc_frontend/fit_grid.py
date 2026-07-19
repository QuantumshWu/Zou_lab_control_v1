"""Compact, headless presentation values for exact saved fit grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math

from zlc_data import (
    REPEAT,
    AxisLayout,
    AxisSpec,
    FitBatchStatus,
    FitResultBatch,
    IndexRangeSelection,
    IndexSelection,
    Selection,
    axis_layout_retained_upper_bound_nbytes,
    axis_spec_retained_upper_bound_nbytes,
    resolve_selection_indices,
)
from zlc_storage import canonical_text, positive_integer

from .figure import RepeatViewMode, ViewPreferences


_GRID_PAGE_CELL_LIMIT = 36
FIT_GRID_EXACT_INDEX_INPUT_MAX_CHARACTERS = 2048
_FIT_GRID_NAVIGATOR_AXIS_MEMORY_BYTES = 128 << 10


def bounded_coordinate_label(value: object) -> str:
    """Return a bounded display label; index authority retains the exact value."""

    if isinstance(value, str):
        if len(value) <= 192:
            return value
        return f"{value[:128]}…{value[-32:]} [length={len(value)}]"
    if isinstance(value, int) and value.bit_length() > 4096:
        sign = "negative " if value < 0 else ""
        return f"<{sign}integer; {value.bit_length()} bits>"
    return str(value)


def _bounded_index_tuple_label(values: tuple[int, ...]) -> str:
    labels = tuple(bounded_coordinate_label(value) for value in values)
    if len(labels) == 1:
        return f"({labels[0]},)"
    return f"({', '.join(labels)})"


def _bounded_coordinate_label_upper_bound_characters(value: object) -> int:
    """Bound :func:`bounded_coordinate_label` without decimalizing a bigint."""

    if isinstance(value, str):
        if len(value) <= 192:
            return len(value)
        # ``prefix…suffix [length=N]``; the decimalized value is only len(value).
        return 176 + len(str(len(value)))
    if isinstance(value, int):
        bits = value.bit_length()
        if bits > 4096:
            return 32 + len(str(bits))
        # ceil(bits * log10(2)) plus sign and conservative punctuation slack.
        return 4 + max(1, (bits * 30103 + 99999) // 100000)
    if isinstance(value, float):
        return 32
    if value is None:
        return 4
    # AxisSpec permits only int/float/str/null coordinates.  Keep this helper
    # fail-closed if an invalid hand-written evaluated DTO reaches the UI.
    raise TypeError("coordinate labels require int/float/str/null values")


def _axis_address(axis: AxisSpec, index: int) -> str:
    coordinate = bounded_coordinate_label(axis.coordinate_at(index))
    index_label = bounded_coordinate_label(index)
    unit = "" if axis.unit is None else f" {axis.unit}"
    return f"{axis.name}={coordinate}{unit} [index {index_label}]"


def _fit_cell_address_upper_bound_characters(
    axes: tuple[AxisSpec, ...],
    multi_index: tuple[int, ...],
) -> int:
    """Bound one logical cell address before allocating any formatted text."""

    if len(axes) != len(multi_index):
        raise ValueError("fit cell address rank differs from its batch axes")
    if not axes:
        return len("scalar batch cell")
    total = 3 * (len(axes) - 1)
    for axis, index in zip(axes, multi_index, strict=True):
        total += (
            len(axis.name)
            + 1
            + _bounded_coordinate_label_upper_bound_characters(
                axis.coordinate_at(index)
            )
            + (0 if axis.unit is None else 1 + len(axis.unit))
            + len(" [index ]")
            + _bounded_coordinate_label_upper_bound_characters(index)
        )
    return total


def _fit_cell_summary_upper_bound_characters(
    result: FitResultBatch,
    storage_index: int | None,
    address_characters: int,
) -> int:
    """Bound one panel summary before its address/lines are materialized."""

    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    if isinstance(address_characters, bool) or not isinstance(
        address_characters, int
    ) or address_characters < 0:
        raise ValueError("address_characters must be a non-negative integer")
    if storage_index is None:
        return address_characters + 512
    if (
        isinstance(storage_index, bool)
        or not isinstance(storage_index, int)
        or not 0 <= storage_index < result.batch_layout.storage_size
    ):
        raise IndexError("fit summary storage_index is outside the saved batch")
    status = result.statuses[storage_index]
    if status is not FitBatchStatus.CONVERGED:
        error = result.errors[storage_index]
        return address_characters + 768 + (0 if error is None else len(error))
    parameter_characters = sum(
        len(definition.name) + 2 * len(unit) + 96
        for definition, unit in zip(
            result.parameter_definitions,
            result.parameter_units,
            strict=True,
        )
    )
    return address_characters + parameter_characters + 1024


def _fit_cell_address(
    axes: tuple[AxisSpec, ...],
    multi_index: tuple[int, ...],
) -> str:
    if len(axes) != len(multi_index):
        raise ValueError("fit cell address rank differs from its batch axes")
    return " | ".join(
        _axis_address(axis, index)
        for axis, index in zip(axes, multi_index, strict=True)
    ) or "scalar batch cell"


def _fit_cell_summary_text(
    result: FitResultBatch,
    storage_index: int,
    address: str,
) -> str:
    """Single formatter for saved-fit focus details in every frontend."""

    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    if (
        isinstance(storage_index, bool)
        or not isinstance(storage_index, int)
        or not 0 <= storage_index < result.batch_layout.storage_size
    ):
        raise IndexError("fit summary storage_index is outside the saved batch")
    canonical_text(address, "fit cell address")
    status = result.statuses[storage_index]
    lines = [
        address,
        f"storage row {storage_index} · status {status.value}",
        (
            "observations present/valid/used "
            f"{int(result.present_observation_counts[storage_index])}/"
            f"{int(result.valid_observation_counts[storage_index])}/"
            f"{int(result.used_observation_counts[storage_index])} · "
            f"evaluations {int(result.evaluation_counts[storage_index])}"
        ),
    ]
    if status is not FitBatchStatus.CONVERGED:
        lines.append("parameters: N/A")
        lines.append(
            result.errors[storage_index] or "fit failed without diagnostic"
        )
        return "\n".join(lines)
    parameter_parts = []
    covariance_valid = bool(result.covariance_valid[storage_index])
    for position, (definition, unit, value) in enumerate(
        zip(
            result.parameter_definitions,
            result.parameter_units,
            result.parameter_values[storage_index],
            strict=True,
        )
    ):
        text = f"{definition.name}={float(value):.7g} {unit}"
        if covariance_valid:
            uncertainty = math.sqrt(
                float(result.covariance[storage_index, position, position])
            )
            text += f" ± {uncertainty:.3g} {unit}"
        parameter_parts.append(text)
    lines.append(" · ".join(parameter_parts))
    used = int(result.used_observation_counts[storage_index])
    rmse = math.sqrt(float(result.residual_sum_squares[storage_index]) / used)
    quality = f"RMSE={rmse:.7g}"
    if result.r_squared_valid[storage_index]:
        quality += f" · R²={float(result.r_squared[storage_index]):.7g}"
    else:
        quality += " · R² unavailable"
    quality += (
        " · covariance valid"
        if covariance_valid
        else " · covariance unavailable"
    )
    lines.append(quality)
    return "\n".join(lines)


def _page_spans(axes: tuple[AxisSpec, ...]) -> tuple[int, ...]:
    spans = [1] * len(axes)
    remaining = _GRID_PAGE_CELL_LIMIT
    for position in range(len(axes) - 1, -1, -1):
        span = min(axes[position].size, remaining)
        spans[position] = span
        remaining = max(1, remaining // span)
    if math.prod(spans) > _GRID_PAGE_CELL_LIMIT:
        raise RuntimeError("fit grid page construction exceeded its cell limit")
    return tuple(spans)


def _compact_retained_upper_bound(
    artifact_identity: str,
    model_id: str,
    fit_axes: tuple[AxisSpec, ...],
    axes: tuple[AxisSpec, ...],
    layout: AxisLayout,
) -> int:
    axis_bytes = sum(
        axis_spec_retained_upper_bound_nbytes(axis)
        for axis in (*fit_axes, *axes)
    )
    ui_text_characters = sum(
        len(axis.axis_id.value)
        + len(axis.name)
        + len(axis.role.value)
        + (0 if axis.unit is None else len(axis.unit))
        for axis in (*fit_axes, *axes)
    )
    # The compact model retains the authoritative AxisLayout but no fit
    # arrays, covariance, per-cell error tuple, or formatted row table.
    layout_bytes = axis_layout_retained_upper_bound_nbytes(layout)
    return positive_integer(
        64 * 1024
        + 4 * (len(artifact_identity) + len(model_id))
        + axis_bytes
        + layout_bytes
        # Model summaries, page/cell labels, the Qt navigator and transient
        # handoff strings can coexist.  Coordinate values themselves are
        # bounded by _coordinate_label and remain authoritative by index.
        + 16 * ui_text_characters
        + 2048 * len(FitBatchStatus),
        "fit grid compact retained upper bound",
    )


def fit_grid_model_retained_upper_bound_nbytes(
    artifact_identity: str,
    result: FitResultBatch,
) -> int:
    """Preflight one compact grid model without constructing it or textifying coordinates."""

    canonical_text(artifact_identity, "fit artifact identity")
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    return _compact_retained_upper_bound(
        artifact_identity,
        result.spec.model_id,
        result.fit_axis_specs,
        result.batch_axis_specs,
        result.batch_layout,
    )


def fit_grid_navigation_retained_upper_bound_nbytes(
    result: FitResultBatch,
) -> int:
    """Reserve one current page/cell summary alongside the compact model."""

    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    axes = (*result.fit_axis_specs, *result.batch_axis_specs)
    axis_text = sum(
        len(axis.axis_id.value)
        + len(axis.name)
        + len(axis.role.value)
        + (0 if axis.unit is None else len(axis.unit))
        for axis in axes
    )
    parameter_text = sum(
        len(definition.name) + len(unit)
        for definition, unit in zip(
            result.parameter_definitions,
            result.parameter_units,
            strict=True,
        )
    )
    error_text = max(
        (0 if value is None else len(value) for value in result.errors),
        default=0,
    )
    # Page and focused-cell values are mutually exclusive, but the returned
    # string, Qt copy, and worker handoff can coexist.  Coordinate labels are
    # deliberately bounded by _coordinate_label; authority remains the exact
    # IndexSelection rather than the presentation string.
    return positive_integer(
        64 * 1024
        + 24 * (axis_text + parameter_text + error_text)
        # One navigator editor can transiently retain Qt UTF-16, Python text,
        # parse input, and a Python bigint.  This dominates the shared bounded
        # direct-entry contract while physical Previous/Next remains exact for
        # still larger sparse logical indices.
        + _FIT_GRID_NAVIGATOR_AXIS_MEMORY_BYTES * len(axes),
        "fit grid navigation retained upper bound",
    )


@dataclass(frozen=True, slots=True)
class FitGridPage:
    """One bounded logical tile containing at least one stored fit cell."""

    address: tuple[int, ...]
    selection: Selection | None
    preferences: ViewPreferences | None
    previous_address: tuple[int, ...] | None
    next_address: tuple[int, ...] | None
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "address", tuple(self.address))
        if self.selection is not None and not isinstance(self.selection, Selection):
            raise TypeError("page selection must be Selection or None")
        if self.preferences is not None and not isinstance(
            self.preferences,
            ViewPreferences,
        ):
            raise TypeError("page preferences must be ViewPreferences or None")
        for name in ("previous_address", "next_address"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        canonical_text(self.label, "fit grid page label")


@dataclass(frozen=True, slots=True)
class FitGridCellSummary:
    """Small display-only projection of one stored result row."""

    selection: Selection | None
    storage_index: int
    status: FitBatchStatus
    text: str

    def __post_init__(self) -> None:
        if self.selection is not None and not isinstance(self.selection, Selection):
            raise TypeError("cell summary selection must be Selection or None")
        if not isinstance(self.storage_index, int) or self.storage_index < 0:
            raise ValueError("cell summary storage_index must be non-negative")
        if not isinstance(self.status, FitBatchStatus):
            raise TypeError("cell summary status must be FitBatchStatus")
        canonical_text(self.text, "fit grid cell summary")


@dataclass(frozen=True, slots=True)
class FitGridModel:
    """Compact navigation metadata for one immutable saved fit artifact.

    The model intentionally does not retain ``FitResultBatch``.  Every page,
    focus, and export worker reloads the exact durable reference once, renders
    it, and returns only this compact axis/layout index plus a bounded raster.
    """

    artifact_identity: str
    model_id: str
    fit_axes: tuple[AxisSpec, ...]
    axes: tuple[AxisSpec, ...]
    layout: AxisLayout
    status_counts: tuple[tuple[FitBatchStatus, int], ...]
    page_spans: tuple[int, ...] = field(init=False)
    retained_upper_bound_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        canonical_text(self.artifact_identity, "fit artifact identity")
        canonical_text(self.model_id, "fit model id")
        fit_axes = tuple(self.fit_axes)
        axes = tuple(self.axes)
        if any(not isinstance(axis, AxisSpec) for axis in (*fit_axes, *axes)):
            raise TypeError("fit grid axes must contain AxisSpec values")
        if len({axis.axis_id for axis in (*fit_axes, *axes)}) != len(
            (*fit_axes, *axes)
        ):
            raise ValueError("fit grid axes must have unique AxisId values")
        if not isinstance(self.layout, AxisLayout):
            raise TypeError("fit grid layout must be AxisLayout")
        if self.layout.logical_shape != tuple(axis.size for axis in axes):
            raise ValueError("fit grid layout shape differs from batch axes")
        counts = tuple(self.status_counts)
        if any(
            not isinstance(status, FitBatchStatus)
            or not isinstance(count, int)
            or count < 0
            for status, count in counts
        ):
            raise TypeError("fit grid status_counts contain invalid entries")
        if len({status for status, _count in counts}) != len(counts):
            raise ValueError("fit grid status_counts contain duplicate statuses")
        if sum(count for _status, count in counts) != self.layout.storage_size:
            raise ValueError("fit grid status counts differ from layout storage")
        object.__setattr__(self, "fit_axes", fit_axes)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "status_counts", counts)
        object.__setattr__(self, "page_spans", _page_spans(axes))
        object.__setattr__(
            self,
            "retained_upper_bound_bytes",
            _compact_retained_upper_bound(
                self.artifact_identity,
                self.model_id,
                fit_axes,
                axes,
                self.layout,
            ),
        )

    @classmethod
    def from_result(
        cls,
        artifact_identity: str,
        result: FitResultBatch,
    ) -> "FitGridModel":
        if not isinstance(result, FitResultBatch):
            raise TypeError("result must be FitResultBatch")
        counts = tuple(
            (status, sum(item is status for item in result.statuses))
            for status in FitBatchStatus
            if any(item is status for item in result.statuses)
        )
        return cls(
            artifact_identity,
            result.spec.model_id,
            result.fit_axis_specs,
            result.batch_axis_specs,
            result.batch_layout,
            counts,
        )

    @property
    def identity(self) -> tuple:
        """Small immutable identity used to reject late/substituted worker views."""

        return (
            self.artifact_identity,
            self.model_id,
            self.fit_axes,
            self.axes,
            self.layout,
            self.status_counts,
        )

    @property
    def summary(self) -> str:
        fit_axes = ", ".join(
            f"{axis.name} ({axis.role.value})" for axis in self.fit_axes
        )
        batch_axes = ", ".join(
            f"{axis.name} ({axis.role.value}, "
            f"{bounded_coordinate_label(axis.size)})"
            for axis in self.axes
        ) or "one scalar batch"
        statuses = ", ".join(
            f"{status.value.lower()}={bounded_coordinate_label(count)}"
            for status, count in self.status_counts
        ) or "no stored cells"
        return (
            f"{self.model_id} · fit axes: {fit_axes} · batch axes: {batch_axes} · "
            f"{statuses}"
        )

    def selection_for_indices(
        self,
        indices: tuple[int, ...],
    ) -> Selection | None:
        multi = tuple(indices)
        self.layout.storage_index(multi)
        if not self.axes:
            return None
        return Selection(
            tuple(
                IndexSelection(axis.axis_id, index)
                for axis, index in zip(self.axes, multi, strict=True)
            )
        )

    def resolve_selection(
        self,
        selection: Selection | None,
    ) -> tuple[int, tuple[int, ...], str]:
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        if not self.axes:
            if selection is not None or self.layout.storage_size != 1:
                raise ValueError("scalar fit batch accepts only the implicit cell")
            return 0, (), "scalar batch cell"
        by_axis = {} if selection is None else {
            term.axis_id: term for term in selection.terms
        }
        known = {axis.axis_id for axis in self.axes}
        if any(axis_id not in known for axis_id in by_axis):
            raise ValueError("fit grid selection may name only saved batch axes")
        indices = []
        for axis in self.axes:
            term = by_axis.get(axis.axis_id)
            if term is None:
                if axis.size != 1:
                    raise ValueError(
                        f"fit cell requires an explicit index for axis {axis.axis_id}"
                    )
                index = 0
            else:
                if not isinstance(term, IndexSelection):
                    raise TypeError(
                        "fit cell selection accepts exact IndexSelection terms"
                    )
                resolved, drop = resolve_selection_indices(axis, term)
                if not drop or len(resolved) != 1:
                    raise ValueError("fit cell selection must resolve one exact index")
                index = resolved.start
            indices.append(index)
        multi = tuple(indices)
        try:
            storage = self.layout.storage_index(multi)
        except KeyError as error:
            raise ValueError(
                "selected logical fit cell "
                f"{_bounded_index_tuple_label(multi)} is absent from the saved "
                "batch layout"
            ) from error
        return storage, multi, _fit_cell_address(self.axes, multi)

    def storage_index_or_none(self, selection: Selection | None) -> int | None:
        try:
            storage, _multi, _label = self.resolve_selection(selection)
        except ValueError as error:
            if "absent from the saved batch layout" in str(error):
                return None
            raise
        return storage

    def _page_address(self, multi: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            index // span
            for index, span in zip(multi, self.page_spans, strict=True)
        )

    def page(self, address: tuple[int, ...] | None = None) -> FitGridPage:
        if self.layout.storage_size <= 0:
            raise ValueError("saved fit result contains no batch cells")
        if address is None:
            resolved_address = self._page_address(self.layout.multi_index(0))
        else:
            resolved_address = tuple(address)
            if len(resolved_address) != len(self.axes) or any(
                not isinstance(value, int) or value < 0
                for value in resolved_address
            ):
                raise ValueError("fit grid page address is invalid")
        page_has_storage = False
        previous_address = None
        next_address = None
        for storage in range(self.layout.storage_size):
            candidate = self._page_address(self.layout.multi_index(storage))
            if candidate == resolved_address:
                page_has_storage = True
            elif candidate < resolved_address and (
                previous_address is None or candidate > previous_address
            ):
                previous_address = candidate
            elif candidate > resolved_address and (
                next_address is None or candidate < next_address
            ):
                next_address = candidate
        if not page_has_storage:
            raise ValueError("fit grid page contains no stored batch cells")
        terms = []
        bounds = []
        facet_axis_ids = []
        for axis, span, page_index in zip(
            self.axes,
            self.page_spans,
            resolved_address,
            strict=True,
        ):
            start = page_index * span
            stop = min(axis.size, start + span)
            if start >= axis.size:
                raise ValueError("fit grid page address is outside a batch axis")
            terms.append(IndexRangeSelection(axis.axis_id, start, stop))
            bounds.append(
                f"{axis.name}[{bounded_coordinate_label(start)}:"
                f"{bounded_coordinate_label(stop)}]"
            )
            if axis.role != REPEAT and stop - start > 1:
                facet_axis_ids.append(axis.axis_id)
        selection = None if not terms else Selection(tuple(terms))
        repeat_mode = (
            RepeatViewMode.FACET
            if any(axis.role == REPEAT for axis in self.axes)
            else None
        )
        preferences = (
            None
            if not self.axes
            else ViewPreferences(
                repeat_mode=repeat_mode,
                facet_axis_ids=tuple(facet_axis_ids),
            )
        )
        return FitGridPage(
            resolved_address,
            selection,
            preferences,
            previous_address,
            next_address,
            " · ".join(bounds) or "scalar fit cell",
        )

    def page_logical_selections(
        self,
        page: FitGridPage,
    ) -> tuple[Selection | None, ...]:
        """Return every logical tile cell in deterministic row-major order.

        Unlike :meth:`selection_for_indices`, this intentionally includes
        EXPLICIT-layout holes.  A renderer must preserve those positions as
        ``NOT_PRESENT`` rather than compacting later physical rows forward.
        """

        if not isinstance(page, FitGridPage):
            raise TypeError("page must be FitGridPage")
        if page != self.page(page.address):
            raise ValueError("fit grid page differs from compact model")
        if not self.axes:
            return (None,)
        ranges = tuple(
            range(
                page_index * span,
                min(axis.size, (page_index + 1) * span),
            )
            for axis, span, page_index in zip(
                self.axes,
                self.page_spans,
                page.address,
                strict=True,
            )
        )
        return tuple(
            Selection(
                tuple(
                    IndexSelection(axis.axis_id, index)
                    for axis, index in zip(self.axes, multi, strict=True)
                )
            )
            for multi in product(*ranges)
        )

    def focus_preferences(self) -> ViewPreferences | None:
        if any(axis.role == REPEAT for axis in self.axes):
            return ViewPreferences(repeat_mode=RepeatViewMode.FACET)
        return None

    def cell_summary(
        self,
        result: FitResultBatch,
        selection: Selection | None,
    ) -> FitGridCellSummary:
        if not isinstance(result, FitResultBatch):
            raise TypeError("result must be FitResultBatch")
        if (
            result.spec.model_id != self.model_id
            or result.fit_axis_specs != self.fit_axes
            or result.batch_axis_specs != self.axes
            or result.batch_layout is not self.layout
        ):
            # The exact-ref loader normally shares these immutable values.
            # Refuse any composition callback that substitutes another result.
            raise ValueError("fit cell summary result differs from grid metadata")
        storage, _multi, address = self.resolve_selection(selection)
        status = result.statuses[storage]
        return FitGridCellSummary(
            selection,
            storage,
            status,
            _fit_cell_summary_text(result, storage, address),
        )


__all__ = [
    "FIT_GRID_EXACT_INDEX_INPUT_MAX_CHARACTERS",
    "bounded_coordinate_label",
    "FitGridCellSummary",
    "FitGridModel",
    "FitGridPage",
    "fit_grid_model_retained_upper_bound_nbytes",
    "fit_grid_navigation_retained_upper_bound_nbytes",
]
