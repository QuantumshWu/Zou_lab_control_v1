"""Pure current-only PulseDocument authoring transforms."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Real
from typing import Mapping, Sequence, TypeVar
from uuid import uuid4

from .document import (
    DEFAULT_PERIOD_DURATION,
    DEFAULT_TIME_UNIT,
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
from .target import PORT_DAC, PORT_DIGITAL, PulseTarget


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


def rename_port_label(
    document: PulseDocument,
    port_key: str,
    label: str,
) -> PulseDocument:
    """Replace one presentation label without changing the target ABI.

    A blank edit restores the stable port key, matching the formal editor's
    established display fallback while keeping ``PulseTarget`` canonical.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    key = str(port_key)
    if key not in document.target.by_key:
        raise KeyError(f"unknown pulse port {key!r}")
    normalized = str(label).strip() or key
    ports = tuple(
        replace(port, label=normalized) if port.key == key else port
        for port in document.target.ports
    )
    target = PulseTarget(document.target.raw_lanes, ports)
    if target.abi_fingerprint != document.target.abi_fingerprint:
        raise RuntimeError("a port label edit changed the target ABI")
    return replace(document, target=target)


def allocate_field_parameter_id(
    document: PulseDocument,
    field: PulseFieldRef,
) -> str:
    """Allocate one readable handle, then keep that handle stable."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(field, PulseFieldRef):
        raise TypeError("field must be PulseFieldRef")
    period_suffix = ""
    if field.period_id is not None:
        period_suffix = f"_p{tuple(document.period_by_id).index(field.period_id) + 1}"
    if field.kind == FIELD_DURATION:
        stem = f"duration{period_suffix}"
    elif field.port is not None:
        label = document.target.by_key[field.port].label
        stem = (
            f"{label}{period_suffix}"
            if field.kind == FIELD_DAC
            else f"{label}_delay"
        )
    else:
        stem = "pulse_parameter"
    stem = (
        re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_").lower()
        or "pulse_parameter"
    )
    if stem[0].isdigit():
        stem = f"pulse_{stem}"
    used = {
        parameter.parameter_id
        for parameter in (*document.scan_parameters, *document.api_parameters)
    }
    candidate = stem
    suffix = 2
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    return candidate


def cycle_field_binding(
    document: PulseDocument,
    field: PulseFieldRef,
    *,
    cascade: bool = False,
) -> PulseEditResult:
    """Apply the formal dot cycle using current typed field identities.

    Duration/DAC fields cycle literal -> SCAN -> API -> literal.  Output
    delays are not scannable and cycle literal -> API -> literal.  A DAC Hold
    becomes an explicit Edge at its carried value when first parameterized,
    matching the hardware meaning of changing that period's DAC endpoint.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(field, PulseFieldRef):
        raise TypeError("field must be PulseFieldRef")
    working = document
    old_scan = next(
        (item for item in working.scan_parameters if item.field == field),
        None,
    )
    old_api = next(
        (item for item in working.api_parameters if item.field == field),
        None,
    )
    if old_api is not None:
        return replace_field_binding(
            working,
            field,
            None,
            cascade=cascade,
        )

    if old_scan is not None:
        _value, unit = working.field_value(field)
        return replace_field_binding(
            working,
            field,
            ApiParameter(old_scan.parameter_id, field, unit),
            cascade=cascade,
        )

    if field.kind == FIELD_DELAY:
        if field.port not in {item.port for item in working.delays}:
            created = set_output_delay(
                working,
                field.port,
                OutputDelay(field.port, 0, DEFAULT_TIME_UNIT),
            )
            working = created.document
        _value, unit = working.field_value(field)
        binding = ApiParameter(
            allocate_field_parameter_id(working, field),
            field,
            unit,
        )
        return replace_field_binding(
            working,
            field,
            binding,
            cascade=cascade,
        )

    if field.kind == FIELD_DAC:
        period = working.period_by_id[field.period_id]
        if field.port not in {step.port for step in period.analog_steps}:
            carried = working.effective_dac_cells()[
                (field.period_id, field.port)
            ][1]
            materialized = set_analog_action(
                working,
                field.period_id,
                field.port,
                AnalogStep(field.port, "edge", carried),
            )
            working = materialized.document

    _value, field_unit = working.field_value(field)
    scan_unit = "ns" if field.kind == FIELD_DURATION else field_unit
    label = (
        ""
        if field.port is None
        else working.target.by_key[field.port].label
    )
    binding = ScanParameter(
        allocate_field_parameter_id(working, field),
        field,
        label,
        scan_unit,
    )
    return replace_field_binding(
        working,
        field,
        binding,
        cascade=cascade,
    )


def new_pulse_document(
    target: PulseTarget,
    *,
    time_step_ns: int | float,
    name: str = "Untitled pulse",
    duration: int | float = DEFAULT_PERIOD_DURATION,
    unit: str = DEFAULT_TIME_UNIT,
) -> PulseDocument:
    """Create the one-period, all-safe document used by a new editor session."""

    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    return PulseDocument(
        name=name,
        target=target,
        time_step_ns=time_step_ns,
        periods=(
            PulsePeriod(
                period_id="p1",
                duration=duration,
                unit=unit,
                name="",
                states=tuple(0 for _ in target.raw_lanes),
            ),
        ),
        visible_ports=tuple(
            port.key for port in target.ports if port.kind in (PORT_DIGITAL, PORT_DAC)
        ),
    )


def set_digital_output(
    document: PulseDocument,
    period_id: str,
    port: str,
    high: bool,
) -> PulseDocument:
    """Set one logical digital output without exposing the target lane layout."""

    if not isinstance(high, bool):
        raise TypeError("high must be bool")
    target_port = document.target.by_key.get(port)
    if target_port is None or target_port.kind != PORT_DIGITAL:
        raise ValueError(f"port {port!r} is not a digital output")
    period = _period(document, period_id)
    lane_index = document.target.raw_lanes.index(target_port.lanes[0])
    states = list(period.states)
    states[lane_index] = int(high)
    if tuple(states) == period.states:
        return document
    periods = tuple(
        replace(item, states=tuple(states)) if item.period_id == period_id else item
        for item in document.periods
    )
    return replace(document, periods=periods)


def set_analog_action(
    document: PulseDocument,
    period_id: str,
    port: str,
    action: AnalogStep | None,
    *,
    cascade: bool = False,
) -> PulseEditResult:
    """Add, replace, or remove one DAC action and its dependent binding atomically."""

    target_port = document.target.by_key.get(port)
    if target_port is None or target_port.kind != PORT_DAC:
        raise ValueError(f"port {port!r} is not a DAC output")
    period = _period(document, period_id)
    current = next((item for item in period.analog_steps if item.port == port), None)
    if action is not None:
        if not isinstance(action, AnalogStep):
            raise TypeError("action must be AnalogStep or None")
        if action.port != port:
            raise ValueError("action port differs from the selected DAC output")
        steps = tuple(
            action if item.port == port else item for item in period.analog_steps
        )
        if current is None:
            steps = (*steps, action)
        periods = tuple(
            replace(item, analog_steps=steps) if item.period_id == period_id else item
            for item in document.periods
        )
        return PulseEditResult(replace(document, periods=periods), PulseEditImpact())
    if current is None:
        return PulseEditResult(document, PulseEditImpact())

    unbound = replace_field_binding(
        document,
        PulseFieldRef(FIELD_DAC, period_id, port),
        None,
        cascade=cascade,
    )
    periods = tuple(
        replace(
            item,
            analog_steps=tuple(step for step in item.analog_steps if step.port != port),
        )
        if item.period_id == period_id
        else item
        for item in unbound.document.periods
    )
    return PulseEditResult(replace(unbound.document, periods=periods), unbound.impact)


def set_output_delay(
    document: PulseDocument,
    port: str,
    delay: OutputDelay | None,
    *,
    cascade: bool = False,
) -> PulseEditResult:
    """Add, replace, or remove one output delay and its dependent API binding."""

    target_port = document.target.by_key.get(port)
    if target_port is None or target_port.kind not in (PORT_DIGITAL, PORT_DAC):
        raise ValueError(f"port {port!r} cannot have an output delay")
    current = next((item for item in document.delays if item.port == port), None)
    if delay is not None:
        if not isinstance(delay, OutputDelay):
            raise TypeError("delay must be OutputDelay or None")
        if delay.port != port:
            raise ValueError("delay port differs from the selected output")
        delays = tuple(delay if item.port == port else item for item in document.delays)
        if current is None:
            delays = (*delays, delay)
        return PulseEditResult(replace(document, delays=delays), PulseEditImpact())
    if current is None:
        return PulseEditResult(document, PulseEditImpact())

    unbound = replace_field_binding(
        document,
        PulseFieldRef(FIELD_DELAY, None, port),
        None,
        cascade=cascade,
    )
    delays = tuple(item for item in unbound.document.delays if item.port != port)
    return PulseEditResult(replace(unbound.document, delays=delays), unbound.impact)


def clear_port(
    document: PulseDocument,
    port: str,
    *,
    cascade: bool = False,
) -> PulseEditResult:
    """Clear one logical editor row as one immutable authoring operation.

    Digital rows become low in every period.  DAC rows lose every authored
    edge/ramp.  The row's output delay is deliberately left alone: the formal
    editor presents delay as a separate field and its established ``Clear``
    action clears the schedule row, not that field.  Any scan/API declarations
    owned by removed DAC actions are reported and removed atomically.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    target_port = document.target.by_key.get(str(port))
    if target_port is None or target_port.kind not in (PORT_DIGITAL, PORT_DAC):
        raise ValueError(f"port {port!r} is not a programmable pulse output")

    working = document
    impacts: list[PulseEditImpact] = []
    if target_port.kind == PORT_DIGITAL:
        for period in working.periods:
            working = set_digital_output(working, period.period_id, target_port.key, False)
    else:
        for period in tuple(working.periods):
            result = set_analog_action(
                working,
                period.period_id,
                target_port.key,
                None,
                cascade=True,
            )
            working = result.document
            impacts.append(result.impact)

    impact = _combine_impacts(impacts)
    if impact.destructive and not cascade:
        raise DestructivePulseEditError(impact)
    return PulseEditResult(working, impact)


def clear_pulse_schedule(
    document: PulseDocument,
    *,
    cascade: bool = False,
) -> PulseEditResult:
    """Reset the schedule to the formal editor's single blank 1 us period.

    Hardware identity, authored presentation labels, visible-row selection,
    document name, and clock stay intact.  Periods, delays, bindings, frozen
    scan data/provenance, and repeat are one authoritative destructive edit.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    impact = PulseEditImpact(
        removed_period_ids=tuple(period.period_id for period in document.periods),
        removed_scan_parameters=tuple(
            parameter.parameter_id for parameter in document.scan_parameters
        ),
        removed_scan_columns=(
            () if document.scan_table is None else tuple(document.scan_table.columns)
        ),
        removed_api_parameters=tuple(
            parameter.parameter_id for parameter in document.api_parameters
        ),
        repeat_before=document.repeat,
        repeat_after=None,
        repeat_members_before=_repeat_members(document.periods, document.repeat),
        repeat_members_after=(),
        scan_provenance_removed=document.scan_recipe is not None,
    )
    if impact.destructive and not cascade:
        raise DestructivePulseEditError(impact)
    blank = new_pulse_document(
        document.target,
        time_step_ns=document.time_step_ns,
        name=document.name,
        duration=1,
        unit="us",
    )
    return PulseEditResult(
        replace(blank, visible_ports=document.visible_ports),
        impact,
    )


def _combine_impacts(impacts: Sequence[PulseEditImpact]) -> PulseEditImpact:
    """Combine disjoint field-edit reports without inventing a second command model."""

    values = tuple(impacts)
    return PulseEditImpact(
        removed_period_ids=tuple(
            item for impact in values for item in impact.removed_period_ids
        ),
        removed_scan_parameters=tuple(
            item for impact in values for item in impact.removed_scan_parameters
        ),
        removed_scan_columns=tuple(
            item for impact in values for item in impact.removed_scan_columns
        ),
        removed_api_parameters=tuple(
            item for impact in values for item in impact.removed_api_parameters
        ),
        repeat_before=next(
            (impact.repeat_before for impact in values if impact.repeat_before is not None),
            None,
        ),
        repeat_after=next(
            (impact.repeat_after for impact in reversed(values) if impact.repeat_after is not None),
            None,
        ),
        repeat_members_before=next(
            (impact.repeat_members_before for impact in values if impact.repeat_members_before),
            (),
        ),
        repeat_members_after=next(
            (
                impact.repeat_members_after
                for impact in reversed(values)
                if impact.repeat_members_after
            ),
            (),
        ),
        scan_provenance_removed=any(
            impact.scan_provenance_removed for impact in values
        ),
    )


def replace_field_binding(
    document: PulseDocument,
    field: PulseFieldRef,
    binding: ScanParameter | ApiParameter | None,
    *,
    cascade: bool = False,
) -> PulseEditResult:
    """Atomically switch one existing field between literal, scan, and API binding."""

    if not isinstance(field, PulseFieldRef):
        raise TypeError("field must be PulseFieldRef")
    if binding is not None and not isinstance(binding, (ScanParameter, ApiParameter)):
        raise TypeError("binding must be ScanParameter, ApiParameter, or None")
    document.field_value(field)
    if binding is not None and binding.field != field:
        raise ValueError("binding field differs from the selected physical field")

    old_scan = next(
        (item for item in document.scan_parameters if item.field == field),
        None,
    )
    old_api = next(
        (item for item in document.api_parameters if item.field == field),
        None,
    )
    if (
        binding is None
        and old_scan is None
        and old_api is None
    ) or (
        isinstance(binding, ScanParameter) and binding == old_scan
    ) or (
        isinstance(binding, ApiParameter) and binding == old_api
    ):
        return PulseEditResult(document, PulseEditImpact())

    scan_parameters = _replace_binding_member(
        document.scan_parameters,
        old_scan,
        binding if isinstance(binding, ScanParameter) else None,
    )
    api_parameters = _replace_binding_member(
        document.api_parameters,
        old_api,
        binding if isinstance(binding, ApiParameter) else None,
    )
    scan_table = _replace_scan_column(document, old_scan, binding)
    scan_definition_changed = _scan_binding_identity(old_scan) != _scan_binding_identity(
        binding if isinstance(binding, ScanParameter) else None
    )
    recipe_removed = document.scan_recipe is not None and scan_definition_changed
    recipe = None if recipe_removed else document.scan_recipe

    replacement_scan = binding if isinstance(binding, ScanParameter) else None
    replacement_api = binding if isinstance(binding, ApiParameter) else None
    impact = PulseEditImpact(
        removed_scan_parameters=(
            (old_scan.parameter_id,)
            if old_scan is not None
            and (
                replacement_scan is None
                or replacement_scan.parameter_id != old_scan.parameter_id
            )
            else ()
        ),
        removed_scan_columns=(
            (old_scan.parameter_id,)
            if old_scan is not None
            and document.scan_table is not None
            and (
                replacement_scan is None
                or replacement_scan.parameter_id != old_scan.parameter_id
            )
            else ()
        ),
        removed_api_parameters=(
            (old_api.parameter_id,)
            if old_api is not None
            and (
                replacement_api is None
                or replacement_api.parameter_id != old_api.parameter_id
            )
            else ()
        ),
        scan_provenance_removed=recipe_removed,
    )
    if impact.destructive and not cascade:
        raise DestructivePulseEditError(impact)
    return PulseEditResult(
        replace(
            document,
            scan_parameters=scan_parameters,
            scan_table=scan_table,
            scan_recipe=recipe,
            api_parameters=api_parameters,
        ),
        impact,
    )


_Binding = TypeVar("_Binding", ScanParameter, ApiParameter)


def _period(document: PulseDocument, period_id: str) -> PulsePeriod:
    try:
        return document.period_by_id[period_id]
    except KeyError as exc:
        raise KeyError(f"unknown period {period_id!r}") from exc


def _replace_binding_member(
    values: tuple[_Binding, ...],
    old: _Binding | None,
    new: _Binding | None,
) -> tuple[_Binding, ...]:
    if old is None:
        return values if new is None else (*values, new)
    return tuple(
        new if item == old and new is not None else item
        for item in values
        if item != old or new is not None
    )


def _scan_binding_identity(
    binding: ScanParameter | None,
) -> tuple[str, PulseFieldRef, str] | None:
    if binding is None:
        return None
    return binding.parameter_id, binding.field, binding.unit


def _replace_scan_column(
    document: PulseDocument,
    old: ScanParameter | None,
    binding: ScanParameter | ApiParameter | None,
) -> FrozenScanTable | None:
    table = document.scan_table
    if table is None:
        return None
    new = binding if isinstance(binding, ScanParameter) else None
    columns = list(table.columns)
    rows = [list(row) for row in table.rows]
    if old is not None:
        index = columns.index(old.parameter_id)
        if new is None:
            columns.pop(index)
            for row in rows:
                row.pop(index)
        else:
            columns[index] = new.parameter_id
            if old.unit != new.unit:
                for row in rows:
                    row[index] = _convert_binding_value(
                        row[index],
                        old.field.kind,
                        old.unit,
                        new.unit,
                    )
    elif new is not None:
        nominal, nominal_unit = document.field_value(new.field)
        value = _convert_binding_value(
            nominal,
            new.field.kind,
            nominal_unit,
            new.unit,
        )
        columns.append(new.parameter_id)
        for row in rows:
            row.append(value)
    if not columns:
        return None
    return FrozenScanTable(tuple(columns), tuple(tuple(row) for row in rows))


def _convert_binding_value(
    value: int | float,
    field_kind: str,
    source_unit: str,
    target_unit: str,
) -> int | float:
    if field_kind not in (FIELD_DURATION, FIELD_DELAY) or source_unit == target_unit:
        return value
    converted = (
        Fraction(str(value))
        * Fraction(str(TIME_UNIT_TO_NS[source_unit]))
        / Fraction(str(TIME_UNIT_TO_NS[target_unit]))
    )
    return converted.numerator if converted.denominator == 1 else float(converted)


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
    """Resolve supplied host-side API values and remove their declarations."""

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
    return replace(
        resolved,
        api_parameters=tuple(
            parameter
            for parameter in resolved.api_parameters
            if parameter.parameter_id not in requested
        ),
    )


def apply_api_values_to_authoring_document(
    document: PulseDocument,
    values: Mapping[str, int | float],
) -> PulseDocument:
    """Apply exact API values while retaining the source authoring form.

    API values use each parameter's declared unit.  A source field may use a
    different equivalent time unit, so convert back before replacing it.  The
    result is the exact applied authoring intent with API declarations intact;
    execution lowering remains :func:`resolve_api_parameters`' responsibility.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    requested = dict(values)
    expected = set(document.api_parameter_by_id)
    if set(requested) != expected:
        raise ValueError(
            "applied API values must exactly cover declared parameters; "
            f"missing={tuple(sorted(expected - set(requested)))}, "
            f"unknown={tuple(sorted(set(requested) - expected))}"
        )
    applied = document
    for parameter in document.api_parameters:
        _nominal, authored_unit = document.field_value(parameter.field)
        value = _convert_binding_value(
            requested[parameter.parameter_id],
            parameter.field.kind,
            parameter.unit,
            authored_unit,
        )
        applied = replace_pulse_field(
            applied,
            parameter.field,
            value,
            unit=authored_unit,
        )
    return applied


def resolve_api_segment_document(
    document: PulseDocument,
    values: Mapping[str, int | float],
) -> PulseDocument:
    """Freeze one finite API-segment pulse without changing physical values.

    API-segment execution has a host boundary between rows, so every row must
    bind the complete declared API domain.  SCAN_SLOT declarations describe a
    different execution mechanism; their already-present document field values
    are the nominal/reference values for this finite segment and the declarations
    are removed.  The authored ``RepeatRegion`` remains the pulse-timeline loop
    inside each point; a scan application's Dataset sweep count is independent.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    requested = dict(values)
    expected_ids = tuple(
        parameter.parameter_id for parameter in document.api_parameters
    )
    expected_set = set(expected_ids)
    if set(requested) != expected_set or len(requested) != len(expected_ids):
        missing = tuple(item for item in expected_ids if item not in requested)
        unknown = tuple(item for item in requested if item not in expected_set)
        raise ValueError(
            "API segment must bind every declared API parameter exactly once; "
            f"missing={missing}, unknown={unknown}"
        )

    finite = replace(
        document,
        scan_parameters=(),
        scan_table=None,
        scan_recipe=None,
    )
    resolved = resolve_api_parameters(finite, requested)
    if resolved.api_parameters:
        raise RuntimeError("API segment retained unresolved API declarations")
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
    duration: int | float = DEFAULT_PERIOD_DURATION,
    unit: str = DEFAULT_TIME_UNIT,
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
    "apply_api_values_to_authoring_document",
    "attach_scan_recipe",
    "allocate_field_parameter_id",
    "clear_port",
    "clear_pulse_schedule",
    "cycle_field_binding",
    "freeze_scan_table",
    "insert_period",
    "inspect_remove_period",
    "move_period",
    "new_pulse_document",
    "new_period",
    "rename_port_label",
    "replace_field_binding",
    "remove_period",
    "replace_pulse_field",
    "resolve_api_segment_document",
    "resolve_api_parameters",
    "set_analog_action",
    "set_digital_output",
    "set_output_delay",
]
