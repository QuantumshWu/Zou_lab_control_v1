"""Pure current-only PulseDocument authoring transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Real
from typing import Mapping, Sequence
from uuid import uuid4

from .document import (
    FIELD_DAC,
    FIELD_DELAY,
    FIELD_DURATION,
    SCAN_NORMALIZER_ID,
    TIME_UNIT_TO_NS,
    AnalogStep,
    ApiParameter,
    FrozenScanTable,
    OutputDelay,
    PulseDocument,
    PulseFieldRef,
    PulsePeriod,
    RepeatRegion,
    ScanParameter,
    ScanRecipeProvenance,
    _exact_ticks,
    _number,
    _time_tick_ratio,
)


@dataclass(frozen=True)
class ScanParameterNormalization:
    parameter_id: str
    unit: str
    adjusted_cells: int
    max_abs_adjustment: float

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_id, str) or not self.parameter_id:
            raise ValueError("parameter_id must be non-empty text")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("unit must be non-empty text")
        if self.adjusted_cells < 0:
            raise ValueError("adjusted_cells must be non-negative")
        if not math.isfinite(self.max_abs_adjustment) or self.max_abs_adjustment < 0:
            raise ValueError("max_abs_adjustment must be finite and non-negative")


@dataclass(frozen=True)
class ScanNormalizationReport:
    parameters: tuple[ScanParameterNormalization, ...]

    def __post_init__(self) -> None:
        values = tuple(self.parameters)
        if any(not isinstance(item, ScanParameterNormalization) for item in values):
            raise TypeError(
                "normalization report parameters must contain ScanParameterNormalization"
            )
        if len({item.parameter_id for item in values}) != len(values):
            raise ValueError("normalization report parameter IDs must be unique")
        object.__setattr__(self, "parameters", values)

    @property
    def adjusted_cells(self) -> int:
        return sum(item.adjusted_cells for item in self.parameters)

    @property
    def by_parameter(self) -> dict[str, ScanParameterNormalization]:
        return {item.parameter_id: item for item in self.parameters}


@dataclass(frozen=True)
class PulseEditImpact:
    removed_period_ids: tuple[str, ...] = ()
    removed_scan_parameters: tuple[str, ...] = ()
    removed_scan_columns: tuple[str, ...] = ()
    removed_api_parameters: tuple[str, ...] = ()
    repeat_before: RepeatRegion | None = None
    repeat_after: RepeatRegion | None = None
    repeat_members_before: tuple[str, ...] = ()
    repeat_members_after: tuple[str, ...] = ()
    scan_provenance_removed: bool = False

    @property
    def destructive(self) -> bool:
        return bool(
            self.removed_scan_parameters
            or self.removed_scan_columns
            or self.removed_api_parameters
            or self.repeat_before != self.repeat_after
            or self.repeat_members_before != self.repeat_members_after
            or self.scan_provenance_removed
        )


@dataclass(frozen=True)
class PulseEditResult:
    document: PulseDocument
    impact: PulseEditImpact


class DestructivePulseEditError(ValueError):
    def __init__(self, impact: PulseEditImpact) -> None:
        self.impact = impact
        super().__init__("pulse edit has cascading authoritative effects; pass cascade=True")


def freeze_scan_table(
    document: PulseDocument,
    columns: Sequence[str],
    raw_rows: Sequence[Sequence[float]],
) -> tuple[FrozenScanTable, ScanNormalizationReport]:
    """Normalize authoring rows to the exact values the frozen target can play."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    column_ids = tuple(str(item) for item in columns)
    if len(set(column_ids)) != len(column_ids):
        raise ValueError("scan columns must be unique")
    parameters = document.scan_parameter_by_id
    if set(column_ids) != set(parameters) or len(column_ids) != len(parameters):
        raise ValueError("scan columns must identify every document parameter exactly once")
    adjusted = {parameter_id: 0 for parameter_id in column_ids}
    maximum = {parameter_id: 0.0 for parameter_id in column_ids}
    normalized_rows: list[tuple[int | float, ...]] = []
    for row_index, row in enumerate(raw_rows):
        values = tuple(row)
        if len(values) != len(column_ids):
            raise ValueError(f"scan row {row_index} width differs from columns")
        normalized: list[float] = []
        for parameter_id, raw_value in zip(column_ids, values):
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError("scan values must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("scan values must be finite")
            parameter = parameters[parameter_id]
            if parameter.field.kind == FIELD_DURATION:
                frozen = _freeze_time_value(
                    document,
                    value,
                    parameter.unit,
                    positive=True,
                )
            else:
                port = document.target.by_key[parameter.field.port]
                assert port.signed_range is not None
                frozen = _round_ties_away(value)
                if not port.signed_range[0] <= frozen <= port.signed_range[1]:
                    raise ValueError(
                        f"scan parameter {parameter_id!r} exceeds DAC range"
                    )
            difference = abs(float(frozen) - value)
            if difference != 0.0:
                adjusted[parameter_id] += 1
                maximum[parameter_id] = max(maximum[parameter_id], difference)
            normalized.append(frozen)
        normalized_rows.append(tuple(normalized))
    return (
        FrozenScanTable(column_ids, tuple(normalized_rows)),
        ScanNormalizationReport(
            tuple(
                ScanParameterNormalization(
                    parameter_id,
                    parameters[parameter_id].unit,
                    adjusted[parameter_id],
                    maximum[parameter_id],
                )
                for parameter_id in column_ids
            )
        ),
    )


def attach_scan_recipe(
    document: PulseDocument,
    *,
    source: str,
    generated_columns: Mapping[str, Sequence[int | float]],
    language: str = "python",
) -> PulseDocument:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if document.scan_table is None:
        raise ValueError("scan recipe requires a frozen scan table")
    values_by_id = dict(generated_columns)
    if set(values_by_id) != set(document.scan_table.columns):
        raise ValueError(
            "scan recipe output must identify every frozen ParameterId exactly once"
        )
    column_values = tuple(
        tuple(values_by_id[parameter_id])
        for parameter_id in document.scan_table.columns
    )
    row_counts = {len(values) for values in column_values}
    if len(row_counts) != 1 or not row_counts or next(iter(row_counts)) < 1:
        raise ValueError("scan recipe output columns must have one shared non-zero length")
    generated_raw_rows = tuple(zip(*column_values))
    regenerated, _report = freeze_scan_table(
        document,
        document.scan_table.columns,
        generated_raw_rows,
    )
    if regenerated.fingerprint != document.scan_table.fingerprint:
        raise ValueError(
            "scan recipe output does not reproduce the current frozen scan table"
        )
    recipe = ScanRecipeProvenance(
        language=language,
        source=source,
        columns=document.scan_table.columns,
        normalizer_id=SCAN_NORMALIZER_ID,
        frozen_definition_digest=document.scan_definition_digest,
    )
    return replace(document, scan_recipe=recipe)


def replace_pulse_field(
    document: PulseDocument,
    reference: PulseFieldRef,
    value: int | float,
    *,
    unit: str | None = None,
) -> PulseDocument:
    """Replace exactly one typed field and return a fully validated document."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(reference, PulseFieldRef):
        raise TypeError("reference must be PulseFieldRef")
    _old_value, old_unit = document.field_value(reference)
    target_unit = old_unit if unit is None else str(unit)
    if reference.kind == FIELD_DURATION:
        frozen = _require_exact_time_value(
            document,
            value,
            target_unit,
            positive=True,
        )
        periods = tuple(
            replace(period, duration=frozen, unit=target_unit)
            if period.period_id == reference.period_id
            else period
            for period in document.periods
        )
        return replace(document, periods=periods)
    if reference.kind == FIELD_DELAY:
        frozen = _require_exact_time_value(
            document,
            value,
            target_unit,
            positive=False,
        )
        delays = tuple(
            replace(delay, value=frozen, unit=target_unit)
            if delay.port == reference.port
            else delay
            for delay in document.delays
        )
        return replace(document, delays=delays)
    if target_unit != "value":
        raise ValueError("DAC field replacements require unit 'value'")
    code = _integral_value(value, "DAC field value")
    port = document.target.by_key[reference.port]
    assert port.signed_range is not None
    if not port.signed_range[0] <= code <= port.signed_range[1]:
        raise ValueError("DAC field value exceeds target range")
    periods = []
    for period in document.periods:
        if period.period_id != reference.period_id:
            periods.append(period)
            continue
        steps = tuple(
            replace(step, value=code) if step.port == reference.port else step
            for step in period.analog_steps
        )
        periods.append(replace(period, analog_steps=steps))
    return replace(document, periods=tuple(periods))


def resolve_api_parameters(
    document: PulseDocument,
    values: Mapping[str, int | float],
) -> PulseDocument:
    """Resolve named host-side API values into a new literal PulseDocument."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    requested = dict(values)
    unknown = set(requested) - set(document.api_parameter_by_id)
    if unknown:
        raise KeyError(f"unknown pulse API parameters: {sorted(unknown)}")
    resolved = document
    for parameter in document.api_parameters:
        if parameter.parameter_id in requested:
            resolved = replace_pulse_field(
                resolved,
                parameter.field,
                requested[parameter.parameter_id],
                unit=parameter.unit,
            )
    return resolved


def _map_field_ref_ports(
    reference: PulseFieldRef,
    mapping: Mapping[str, str],
) -> PulseFieldRef:
    if not isinstance(reference, PulseFieldRef):
        raise TypeError("reference must be PulseFieldRef")
    if reference.port is None:
        return reference
    try:
        rebound = mapping[reference.port]
    except KeyError as exc:
        raise KeyError(f"field references unmapped port {reference.port!r}") from exc
    return replace(reference, port=rebound)


def _new_period_id(document: PulseDocument) -> str:
    used = {period.period_id for period in document.periods}
    while True:
        candidate = f"p_{uuid4().hex}"
        if candidate not in used:
            return candidate


def new_period(
    document: PulseDocument,
    *,
    duration: int | float = 1000,
    unit: str = "ns",
    name: str = "",
    states: Sequence[int] | None = None,
    analog_steps: Sequence[AnalogStep] = (),
) -> PulsePeriod:
    if states is None:
        states = tuple(0 for _ in document.target.raw_lanes)
    frozen = _require_exact_time_value(
        document,
        duration,
        unit,
        positive=True,
    )
    return PulsePeriod(
        period_id=_new_period_id(document),
        duration=frozen,
        unit=unit,
        name=name,
        states=tuple(states),
        analog_steps=tuple(analog_steps),
    )


def insert_period(
    document: PulseDocument,
    *,
    period: PulsePeriod,
    before: str | None = None,
    cascade: bool = False,
) -> PulseEditResult:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(period, PulsePeriod):
        raise TypeError("period must be PulsePeriod")
    if period.period_id in document.period_by_id:
        raise ValueError(f"period_id {period.period_id!r} already exists")
    periods = list(document.periods)
    if before is None:
        periods.append(period)
    else:
        try:
            index = next(
                index for index, item in enumerate(periods) if item.period_id == before
            )
        except StopIteration as exc:
            raise KeyError(f"unknown insertion anchor period {before!r}") from exc
        periods.insert(index, period)
    result = replace(document, periods=tuple(periods))
    impact = PulseEditImpact(
        repeat_before=document.repeat,
        repeat_after=result.repeat,
        repeat_members_before=_repeat_members(document.periods, document.repeat),
        repeat_members_after=_repeat_members(result.periods, result.repeat),
    )
    if impact.destructive and not cascade:
        raise DestructivePulseEditError(impact)
    return PulseEditResult(result, impact)


def move_period(
    document: PulseDocument,
    *,
    period_id: str,
    before: str | None,
    cascade: bool = False,
) -> PulseEditResult:
    if period_id not in document.period_by_id:
        raise KeyError(f"unknown period {period_id!r}")
    if before == period_id:
        return PulseEditResult(document, PulseEditImpact())
    periods = [item for item in document.periods if item.period_id != period_id]
    moved = document.period_by_id[period_id]
    if before is None:
        periods.append(moved)
    else:
        try:
            index = next(
                index for index, item in enumerate(periods) if item.period_id == before
            )
        except StopIteration as exc:
            raise KeyError(f"unknown move anchor period {before!r}") from exc
        periods.insert(index, moved)
    result = replace(document, periods=tuple(periods))
    impact = PulseEditImpact(
        repeat_before=document.repeat,
        repeat_after=result.repeat,
        repeat_members_before=_repeat_members(document.periods, document.repeat),
        repeat_members_after=_repeat_members(result.periods, result.repeat),
    )
    if impact.destructive and not cascade:
        raise DestructivePulseEditError(impact)
    return PulseEditResult(result, impact)


def inspect_remove_period(
    document: PulseDocument,
    period_id: str,
) -> PulseEditImpact:
    if period_id not in document.period_by_id:
        raise KeyError(f"unknown period {period_id!r}")
    if len(document.periods) <= 1:
        raise ValueError("a pulse document must retain at least one period")
    removed_scan = tuple(
        parameter.parameter_id
        for parameter in document.scan_parameters
        if parameter.field.period_id == period_id
    )
    removed_api = tuple(
        parameter.parameter_id
        for parameter in document.api_parameters
        if parameter.field.period_id == period_id
    )
    removed_columns = (
        tuple(
            column
            for column in document.scan_table.columns
            if column in set(removed_scan)
        )
        if document.scan_table is not None
        else ()
    )
    repeat_after = _repeat_after_removal(document, period_id)
    remaining_periods = tuple(
        period for period in document.periods if period.period_id != period_id
    )
    return PulseEditImpact(
        removed_period_ids=(period_id,),
        removed_scan_parameters=removed_scan,
        removed_scan_columns=removed_columns,
        removed_api_parameters=removed_api,
        repeat_before=document.repeat,
        repeat_after=repeat_after,
        repeat_members_before=_repeat_members(document.periods, document.repeat),
        repeat_members_after=_repeat_members(remaining_periods, repeat_after),
        scan_provenance_removed=bool(removed_columns and document.scan_recipe is not None),
    )


def remove_period(
    document: PulseDocument,
    period_id: str,
    *,
    cascade: bool = False,
) -> PulseEditResult:
    impact = inspect_remove_period(document, period_id)
    if impact.destructive and not cascade:
        raise DestructivePulseEditError(impact)
    removed_scan = set(impact.removed_scan_parameters)
    periods = tuple(
        period for period in document.periods if period.period_id != period_id
    )
    scan_parameters = tuple(
        parameter
        for parameter in document.scan_parameters
        if parameter.parameter_id not in removed_scan
    )
    table = document.scan_table
    if table is not None and removed_scan:
        kept_indices = [
            index for index, column in enumerate(table.columns) if column not in removed_scan
        ]
        if kept_indices:
            table = FrozenScanTable(
                tuple(table.columns[index] for index in kept_indices),
                tuple(
                    tuple(row[index] for index in kept_indices) for row in table.rows
                ),
            )
        else:
            table = None
    recipe = None if impact.scan_provenance_removed else document.scan_recipe
    api_parameters = tuple(
        parameter
        for parameter in document.api_parameters
        if parameter.parameter_id not in set(impact.removed_api_parameters)
    )
    result = replace(
        document,
        periods=periods,
        scan_parameters=scan_parameters,
        scan_table=table,
        scan_recipe=recipe,
        api_parameters=api_parameters,
        repeat=impact.repeat_after,
    )
    return PulseEditResult(result, impact)


def _repeat_after_removal(
    document: PulseDocument,
    period_id: str,
) -> RepeatRegion | None:
    repeat = document.repeat
    if repeat is None:
        return None
    ids = [period.period_id for period in document.periods]
    start = ids.index(repeat.start_period_id)
    end = ids.index(repeat.end_period_id)
    removed = ids.index(period_id)
    if removed < start or removed > end:
        return repeat
    remaining = [item for item in ids[start : end + 1] if item != period_id]
    if not remaining:
        return None
    return RepeatRegion(remaining[0], remaining[-1], repeat.count)


def _repeat_members(
    periods: Sequence[PulsePeriod],
    repeat: RepeatRegion | None,
) -> tuple[str, ...]:
    if repeat is None:
        return ()
    ids = tuple(period.period_id for period in periods)
    start = ids.index(repeat.start_period_id)
    end = ids.index(repeat.end_period_id)
    return ids[start : end + 1]


def _freeze_time_value(
    document: PulseDocument,
    value: int | float,
    unit: str,
    *,
    positive: bool,
) -> int | float:
    if unit not in TIME_UNIT_TO_NS:
        raise ValueError(f"unsupported time unit {unit!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("time value must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError("time value must be finite")
    raw_ticks = _time_tick_ratio(value, unit, document.time_step_ns)
    ticks = _round_ties_away(raw_ticks)
    if positive and ticks < 1:
        raise ValueError("time value must be at least one target clock tick")
    frozen = _number(
        ticks * document.time_step_ns / TIME_UNIT_TO_NS[unit],
        "frozen time value",
    )
    _exact_ticks(
        frozen,
        unit,
        document.time_step_ns,
        "frozen time value",
        minimum=1 if positive else None,
    )
    return frozen


def _require_exact_time_value(
    document: PulseDocument,
    value: int | float,
    unit: str,
    *,
    positive: bool,
) -> int | float:
    if unit not in TIME_UNIT_TO_NS:
        raise ValueError(f"unsupported time unit {unit!r}")
    canonical = _number(value, "time value")
    _exact_ticks(
        canonical,
        unit,
        document.time_step_ns,
        "time value",
        minimum=1 if positive else None,
    )
    return canonical


def _integral_value(value: int | float, field: str) -> int:
    numeric = _number(value, field)
    if not isinstance(numeric, int):
        raise ValueError(f"{field} must be integral")
    return numeric


def _round_ties_away(value: object) -> int:
    ratio = (
        value
        if isinstance(value, Fraction)
        else Fraction(str(_number(value, "rounding value")))
    )
    numerator = ratio.numerator
    denominator = ratio.denominator
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return sign * quotient


__all__ = [
    "DestructivePulseEditError",
    "PulseEditImpact",
    "PulseEditResult",
    "ScanParameterNormalization",
    "ScanNormalizationReport",
    "attach_scan_recipe",
    "freeze_scan_table",
    "insert_period",
    "inspect_remove_period",
    "move_period",
    "new_period",
    "remove_period",
    "replace_pulse_field",
    "resolve_api_parameters",
]
