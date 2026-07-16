"""Immutable current pulse authoring model and strict current-only codec.

Authoring uses stable identities and typed field references.  Dense ``sN`` scan
variables are a compiler-lowering detail and can never appear in a saved pulse
document, so the UI target and the physical field cannot diverge.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from zlc_storage import (
    canonical_digest,
    canonical_text as _text,
    integer as _integer,
    sha256_text,
)

from .target import (
    PORT_DAC,
    PORT_DIGITAL,
    PulseTarget,
    pulse_target_from_tree,
    pulse_target_to_tree,
)


PULSE_DOCUMENT_SCHEMA = "zlc_pulse.PulseDocument"
SCAN_TABLE_SCHEMA = "zlc_pulse.FrozenScanTable"
SCAN_NORMALIZER_ID = "zlc-pulse-scan-freeze"
TIME_UNITS = frozenset(("s", "ms", "us", "ns"))
TIME_UNIT_TO_NS = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}
FIELD_DURATION = "duration"
FIELD_DAC = "dac"
FIELD_DELAY = "delay"
FIELD_KINDS = frozenset((FIELD_DURATION, FIELD_DAC, FIELD_DELAY))
SCAN_FIELD_KINDS = frozenset((FIELD_DURATION, FIELD_DAC))
ANALOG_MODES = frozenset(("edge", "ramp"))
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _identifier(value: object, field: str) -> str:
    result = _text(value, field)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{field} must be an identifier")
    return result


def _number(value: object, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    if isinstance(value, int) or float(value).is_integer():
        return int(value)
    return float(value)


def _time_unit(value: object, field: str) -> str:
    unit = _text(value, field)
    if unit not in TIME_UNITS:
        raise ValueError(f"{field} is unsupported")
    return unit


def _integral_code(value: object, field: str) -> int:
    number = float(_number(value, field))
    rounded = round(number)
    if not math.isclose(number, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{field} must be an integral DAC code")
    return int(rounded)


def _time_tick_ratio(value: int | float, unit: str, step_ns: float) -> Fraction:
    return (
        Fraction(str(_number(value, "time value")))
        * Fraction(str(TIME_UNIT_TO_NS[unit]))
        / Fraction(str(_number(step_ns, "time step")))
    )


def _exact_ticks(
    value: int | float,
    unit: str,
    step_ns: float,
    field: str,
    *,
    minimum: int | None = 1,
) -> int:
    raw = _time_tick_ratio(value, unit, step_ns)
    if raw.denominator != 1:
        raise ValueError(f"{field} is not frozen to the document clock grid")
    ticks = raw.numerator
    if minimum is not None and ticks < minimum:
        raise ValueError(f"{field} must be at least one target clock tick")
    return ticks


@dataclass(frozen=True)
class PulseFieldRef:
    """One typed, stable reference to a physical authoring field."""

    kind: str
    period_id: str | None = None
    port: str | None = None

    def __post_init__(self) -> None:
        kind = _text(self.kind, "pulse field kind")
        if kind not in FIELD_KINDS:
            raise ValueError(f"unsupported pulse field kind {kind!r}")
        period_id = self.period_id
        port = self.port
        if kind == FIELD_DURATION:
            if period_id is None or port is not None:
                raise ValueError("duration fields require only period_id")
            period_id = _identifier(period_id, "duration field period_id")
        elif kind == FIELD_DAC:
            if period_id is None or port is None:
                raise ValueError("DAC fields require period_id and port")
            period_id = _identifier(period_id, "DAC field period_id")
            port = _text(port, "DAC field port")
        else:
            if period_id is not None or port is None:
                raise ValueError("delay fields require only port")
            port = _text(port, "delay field port")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "period_id", period_id)
        object.__setattr__(self, "port", port)


@dataclass(frozen=True)
class AnalogStep:
    """One explicit DAC action; absence from a period means hold."""

    port: str
    mode: str
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", _text(self.port, "analog step port"))
        mode = _text(self.mode, "analog step mode")
        if mode not in ANALOG_MODES:
            raise ValueError(f"unsupported analog step mode {mode!r}")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "value", _integral_code(self.value, "analog step value"))


@dataclass(frozen=True)
class PulsePeriod:
    period_id: str
    duration: int | float
    unit: str
    name: str
    states: tuple[int, ...]
    analog_steps: tuple[AnalogStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "period_id", _identifier(self.period_id, "period_id"))
        duration = _number(self.duration, "period duration")
        if float(duration) <= 0:
            raise ValueError("period duration must be positive")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "unit", _time_unit(self.unit, "period unit"))
        object.__setattr__(self, "name", _text(self.name, "period name", empty=True))
        states = tuple(self.states)
        if any(type(value) not in (bool, int) or int(value) not in (0, 1) for value in states):
            raise ValueError("period states must contain only boolean/0/1 values")
        object.__setattr__(self, "states", tuple(int(bool(value)) for value in states))
        steps = tuple(self.analog_steps)
        if any(not isinstance(step, AnalogStep) for step in steps):
            raise TypeError("period analog_steps must contain AnalogStep values")
        if len({step.port for step in steps}) != len(steps):
            raise ValueError("a period can contain at most one analog step per port")
        object.__setattr__(self, "analog_steps", steps)


@dataclass(frozen=True)
class ScanParameter:
    parameter_id: str
    field: PulseFieldRef
    label: str
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_id",
            _identifier(self.parameter_id, "scan parameter_id"),
        )
        if not isinstance(self.field, PulseFieldRef):
            raise TypeError("scan parameter field must be PulseFieldRef")
        if self.field.kind not in SCAN_FIELD_KINDS:
            raise ValueError("only duration and DAC fields can be scanned")
        object.__setattr__(self, "label", _text(self.label, "scan label", empty=True))
        unit = _text(self.unit, "scan parameter unit")
        if self.field.kind == FIELD_DURATION:
            unit = _time_unit(unit, "duration scan unit")
        elif unit != "value":
            raise ValueError("DAC scan parameters use unit 'value'")
        object.__setattr__(self, "unit", unit)


@dataclass(frozen=True)
class ApiParameter:
    parameter_id: str
    field: PulseFieldRef
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_id",
            _identifier(self.parameter_id, "API parameter_id"),
        )
        if not isinstance(self.field, PulseFieldRef):
            raise TypeError("API parameter field must be PulseFieldRef")
        unit = _text(self.unit, "API parameter unit")
        if self.field.kind in (FIELD_DURATION, FIELD_DELAY):
            unit = _time_unit(unit, "time API unit")
        elif unit != "value":
            raise ValueError("DAC API parameters use unit 'value'")
        object.__setattr__(self, "unit", unit)


@dataclass(frozen=True)
class FrozenScanTable:
    """Clock-normalized scan rows whose columns carry stable ParameterIds."""

    columns: tuple[str, ...]
    rows: tuple[tuple[int | float, ...], ...]

    def __post_init__(self) -> None:
        columns = tuple(_identifier(item, "scan table column") for item in self.columns)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("scan table columns must be unique and non-empty")
        rows = tuple(
            tuple(_number(value, "scan table value") for value in row)
            for row in self.rows
        )
        if not rows:
            raise ValueError("a frozen scan table must contain at least one row")
        if any(len(row) != len(columns) for row in rows):
            raise ValueError("every scan table row must match its named columns")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", rows)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(frozen_scan_table_to_tree(self))


@dataclass(frozen=True)
class ScanRecipeProvenance:
    language: str
    source: str
    columns: tuple[str, ...]
    normalizer_id: str
    frozen_definition_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", _text(self.language, "scan recipe language"))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("scan recipe source must be non-empty text")
        columns = tuple(_identifier(item, "scan recipe column") for item in self.columns)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("scan recipe columns must be unique and non-empty")
        object.__setattr__(self, "columns", columns)
        normalizer = _text(self.normalizer_id, "scan recipe normalizer_id")
        if normalizer != SCAN_NORMALIZER_ID:
            raise ValueError("scan recipe normalizer differs from the current owner")
        object.__setattr__(self, "normalizer_id", normalizer)
        sha256_text(
            self.frozen_definition_digest,
            "scan recipe frozen_definition_digest",
        )


@dataclass(frozen=True)
class OutputDelay:
    port: str
    value: int | float
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", _text(self.port, "delay port"))
        object.__setattr__(self, "value", _number(self.value, "delay value"))
        object.__setattr__(self, "unit", _time_unit(self.unit, "delay unit"))


@dataclass(frozen=True)
class RepeatRegion:
    start_period_id: str
    end_period_id: str
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_period_id",
            _identifier(self.start_period_id, "repeat start_period_id"),
        )
        object.__setattr__(
            self,
            "end_period_id",
            _identifier(self.end_period_id, "repeat end_period_id"),
        )
        object.__setattr__(self, "count", _integer(self.count, "repeat count", minimum=2))


@dataclass(frozen=True)
class PulseDocument:
    name: str
    target: PulseTarget
    time_step_ns: float
    periods: tuple[PulsePeriod, ...]
    scan_parameters: tuple[ScanParameter, ...] = ()
    scan_table: FrozenScanTable | None = None
    scan_recipe: ScanRecipeProvenance | None = None
    api_parameters: tuple[ApiParameter, ...] = ()
    visible_ports: tuple[str, ...] = ()
    delays: tuple[OutputDelay, ...] = ()
    repeat: RepeatRegion | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "pulse document name"))
        if not isinstance(self.target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        step_ns = float(_number(self.time_step_ns, "time_step_ns"))
        if step_ns <= 0:
            raise ValueError("time_step_ns must be positive")
        object.__setattr__(self, "time_step_ns", step_ns)

        periods = tuple(self.periods)
        if not periods or any(not isinstance(period, PulsePeriod) for period in periods):
            raise ValueError("periods must contain PulsePeriod values")
        period_ids = tuple(period.period_id for period in periods)
        if len(set(period_ids)) != len(period_ids):
            raise ValueError("period_id values must be unique")
        lane_owner = {
            lane: port for port in self.target.ports for lane in port.lanes
        }
        port_order = {port.key: index for index, port in enumerate(self.target.ports)}
        normalized_periods: list[PulsePeriod] = []
        for period in periods:
            if len(period.states) != len(self.target.raw_lanes):
                raise ValueError("every period state vector must match target raw lanes")
            _exact_ticks(period.duration, period.unit, step_ns, f"period {period.period_id} duration")
            for lane, state in zip(self.target.raw_lanes, period.states):
                if state and lane_owner[lane].kind != PORT_DIGITAL:
                    raise ValueError(
                        f"period {period.period_id} stores a digital state on non-digital lane {lane!r}"
                    )
            for analog in period.analog_steps:
                port = self.target.by_key.get(analog.port)
                if port is None or port.kind != PORT_DAC:
                    raise ValueError(
                        f"period {period.period_id} analog step references unknown DAC {analog.port!r}"
                    )
                assert port.signed_range is not None
                if not port.signed_range[0] <= analog.value <= port.signed_range[1]:
                    raise ValueError(
                        f"period {period.period_id} analog value is outside {analog.port!r} range"
                    )
            ordered_steps = tuple(
                sorted(period.analog_steps, key=lambda item: port_order[item.port])
            )
            normalized_periods.append(
                period
                if ordered_steps == period.analog_steps
                else replace(period, analog_steps=ordered_steps)
            )
        periods = tuple(normalized_periods)
        object.__setattr__(self, "periods", periods)

        visible = tuple(_text(key, "visible port") for key in self.visible_ports)
        if len(set(visible)) != len(visible) or any(
            key not in self.target.by_key
            or self.target.by_key[key].kind not in (PORT_DIGITAL, PORT_DAC)
            for key in visible
        ):
            raise ValueError(
                "visible_ports must be unique digital/DAC keys in the PulseTarget"
            )
        object.__setattr__(self, "visible_ports", visible)

        delays = tuple(self.delays)
        if any(not isinstance(delay, OutputDelay) for delay in delays):
            raise TypeError("delays must contain OutputDelay values")
        if len({delay.port for delay in delays}) != len(delays):
            raise ValueError("at most one delay may be declared per logical port")
        for delay in delays:
            port = self.target.by_key.get(delay.port)
            if port is None or port.kind not in (PORT_DIGITAL, PORT_DAC):
                raise ValueError(f"delay references unsupported port {delay.port!r}")
            _exact_ticks(
                delay.value,
                delay.unit,
                step_ns,
                f"delay {delay.port!r}",
                minimum=None,
            )
        delays = tuple(sorted(delays, key=lambda item: port_order[item.port]))
        object.__setattr__(self, "delays", delays)

        repeat = self.repeat
        if repeat is not None:
            if not isinstance(repeat, RepeatRegion):
                raise TypeError("repeat must be RepeatRegion or None")
            try:
                start = period_ids.index(repeat.start_period_id)
                end = period_ids.index(repeat.end_period_id)
            except ValueError as exc:
                raise ValueError("repeat region references a missing period_id") from exc
            if end < start:
                raise ValueError("repeat region end precedes its start")

        scan_parameters = tuple(self.scan_parameters)
        if any(not isinstance(item, ScanParameter) for item in scan_parameters):
            raise TypeError("scan_parameters must contain ScanParameter values")
        scan_ids = tuple(item.parameter_id for item in scan_parameters)
        if len(set(scan_ids)) != len(scan_ids):
            raise ValueError("scan parameter_id values must be unique")

        api_parameters = tuple(self.api_parameters)
        if any(not isinstance(item, ApiParameter) for item in api_parameters):
            raise TypeError("api_parameters must contain ApiParameter values")
        api_ids = tuple(item.parameter_id for item in api_parameters)
        if len(set(api_ids)) != len(api_ids):
            raise ValueError("API parameter_id values must be unique")
        if set(scan_ids) & set(api_ids):
            raise ValueError("scan and API parameters share one document-level identity namespace")

        scan_fields = tuple(item.field for item in scan_parameters)
        api_fields = tuple(item.field for item in api_parameters)
        if len(set(scan_fields)) != len(scan_fields):
            raise ValueError("a physical field cannot own multiple scan parameters")
        if len(set(api_fields)) != len(api_fields):
            raise ValueError("a physical field cannot own multiple API parameters")
        if set(scan_fields) & set(api_fields):
            raise ValueError("a physical field cannot be both scan- and API-bound")

        for parameter in (*scan_parameters, *api_parameters):
            self._validate_field_ref(parameter.field)
        object.__setattr__(self, "scan_parameters", scan_parameters)
        object.__setattr__(self, "api_parameters", api_parameters)

        table = self.scan_table
        if table is not None:
            if not isinstance(table, FrozenScanTable):
                raise TypeError("scan_table must be FrozenScanTable or None")
            if set(table.columns) != set(scan_ids) or len(table.columns) != len(scan_ids):
                raise ValueError("scan table columns must identify every scan parameter exactly once")
            by_id = {item.parameter_id: item for item in scan_parameters}
            for row_index, row in enumerate(table.rows):
                for parameter_id, value in zip(table.columns, row):
                    self._validate_frozen_scan_value(
                        by_id[parameter_id],
                        value,
                        field=f"scan row {row_index} column {parameter_id!r}",
                    )
        elif self.scan_recipe is not None:
            raise ValueError("scan recipe provenance requires a frozen scan table")

        recipe = self.scan_recipe
        if recipe is not None:
            if not isinstance(recipe, ScanRecipeProvenance):
                raise TypeError("scan_recipe must be ScanRecipeProvenance or None")
            assert table is not None
            if recipe.columns != table.columns:
                raise ValueError("scan recipe columns differ from the frozen scan table")
            if recipe.frozen_definition_digest != self.scan_definition_digest:
                raise ValueError(
                    "scan recipe digest differs from the frozen scan definition"
                )

    def _validate_field_ref(self, reference: PulseFieldRef) -> None:
        periods = {period.period_id: period for period in self.periods}
        if reference.kind == FIELD_DURATION:
            if reference.period_id not in periods:
                raise ValueError(f"duration field references missing period {reference.period_id!r}")
            return
        if reference.kind == FIELD_DELAY:
            port = self.target.by_key.get(reference.port)
            if port is None or port.kind not in (PORT_DIGITAL, PORT_DAC):
                raise ValueError(f"delay field references unsupported port {reference.port!r}")
            if reference.port not in {delay.port for delay in self.delays}:
                raise ValueError("a delay parameter requires an explicit nominal OutputDelay")
            return
        period = periods.get(reference.period_id)
        port = self.target.by_key.get(reference.port)
        if period is None:
            raise ValueError(f"DAC field references missing period {reference.period_id!r}")
        if port is None or port.kind != PORT_DAC:
            raise ValueError(f"DAC field references unknown DAC port {reference.port!r}")
        if reference.port not in {step.port for step in period.analog_steps}:
            raise ValueError("a DAC parameter must reference an explicit edge/ramp step")

    def _validate_frozen_scan_value(
        self,
        parameter: ScanParameter,
        value: float,
        *,
        field: str,
    ) -> None:
        if parameter.field.kind == FIELD_DURATION:
            _exact_ticks(value, parameter.unit, self.time_step_ns, field)
            return
        port = self.target.by_key[parameter.field.port]
        assert port.signed_range is not None
        code = _integral_code(value, field)
        if not port.signed_range[0] <= code <= port.signed_range[1]:
            raise ValueError(f"{field} is outside DAC range")

    @property
    def fingerprint(self) -> str:
        return canonical_digest(pulse_document_to_tree(self))

    @property
    def period_by_id(self) -> dict[str, PulsePeriod]:
        return {period.period_id: period for period in self.periods}

    @property
    def scan_parameter_by_id(self) -> dict[str, ScanParameter]:
        return {parameter.parameter_id: parameter for parameter in self.scan_parameters}

    @property
    def api_parameter_by_id(self) -> dict[str, ApiParameter]:
        return {parameter.parameter_id: parameter for parameter in self.api_parameters}

    @property
    def scan_definition_digest(self) -> str:
        if self.scan_table is None:
            raise ValueError("a scan definition digest requires a frozen scan table")
        by_id = self.scan_parameter_by_id
        parameters = []
        for parameter_id in self.scan_table.columns:
            parameter = by_id[parameter_id]
            reference = parameter.field
            if reference.port is None:
                physical_port = None
            else:
                port = self.target.by_key[reference.port]
                latch_lanes = (
                    None
                    if port.latch_clock is None
                    else list(self.target.by_key[port.latch_clock].lanes)
                )
                physical_port = {
                    "kind": port.kind,
                    "lanes": list(port.lanes),
                    "bus_index": port.bus_index,
                    "width": port.width,
                    "encoding": port.encoding,
                    "safe_value": port.safe_value,
                    "latch_clock_lanes": latch_lanes,
                }
            parameters.append(
                {
                    "parameter_id": parameter.parameter_id,
                    "field_kind": reference.kind,
                    "period_id": reference.period_id,
                    "physical_port": physical_port,
                    "unit": parameter.unit,
                }
            )
        return canonical_digest(
            {
                "schema": "zlc_pulse.FrozenScanDefinition",
                "time_step_ns": self.time_step_ns,
                "parameters": parameters,
                "table": frozen_scan_table_to_tree(self.scan_table),
            }
        )

    def field_value(self, reference: PulseFieldRef) -> tuple[int | float, str]:
        if not isinstance(reference, PulseFieldRef):
            raise TypeError("reference must be PulseFieldRef")
        self._validate_field_ref(reference)
        if reference.kind == FIELD_DURATION:
            period = self.period_by_id[reference.period_id]
            return period.duration, period.unit
        if reference.kind == FIELD_DELAY:
            delay = next(item for item in self.delays if item.port == reference.port)
            return delay.value, delay.unit
        period = self.period_by_id[reference.period_id]
        analog = next(item for item in period.analog_steps if item.port == reference.port)
        return analog.value, "value"

    def save(self, path: str | Path) -> Path:
        return save_pulse_document(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PulseDocument":
        return load_pulse_document(path)


def frozen_scan_table_to_tree(table: FrozenScanTable) -> dict[str, object]:
    if not isinstance(table, FrozenScanTable):
        raise TypeError("table must be FrozenScanTable")
    return {
        "schema": SCAN_TABLE_SCHEMA,
        "columns": list(table.columns),
        "rows": [list(row) for row in table.rows],
    }


def frozen_scan_table_from_tree(tree: object) -> FrozenScanTable:
    if not isinstance(tree, dict) or set(tree) != {"schema", "columns", "rows"}:
        raise ValueError("FrozenScanTable has an unknown field set")
    if tree["schema"] != SCAN_TABLE_SCHEMA:
        raise ValueError("FrozenScanTable schema differs")
    if not isinstance(tree["columns"], list) or not isinstance(tree["rows"], list):
        raise TypeError("FrozenScanTable columns and rows must be lists")
    rows = tree["rows"]
    if any(not isinstance(row, list) for row in rows):
        raise TypeError("FrozenScanTable rows must contain lists")
    return FrozenScanTable(tuple(tree["columns"]), tuple(tuple(row) for row in rows))


def pulse_document_to_tree(document: PulseDocument) -> dict[str, object]:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    return {
        "schema": PULSE_DOCUMENT_SCHEMA,
        "name": document.name,
        "target": pulse_target_to_tree(document.target),
        "time_step_ns": document.time_step_ns,
        "periods": [_period_to_tree(period) for period in document.periods],
        "scan_parameters": [_scan_parameter_to_tree(item) for item in document.scan_parameters],
        "scan_table": (
            None if document.scan_table is None else frozen_scan_table_to_tree(document.scan_table)
        ),
        "scan_recipe": (
            None if document.scan_recipe is None else _scan_recipe_to_tree(document.scan_recipe)
        ),
        "api_parameters": [_api_parameter_to_tree(item) for item in document.api_parameters],
        "visible_ports": list(document.visible_ports),
        "delays": [_delay_to_tree(item) for item in document.delays],
        "repeat": None if document.repeat is None else _repeat_to_tree(document.repeat),
    }


def pulse_document_from_tree(tree: object) -> PulseDocument:
    fields = {
        "schema",
        "name",
        "target",
        "time_step_ns",
        "periods",
        "scan_parameters",
        "scan_table",
        "scan_recipe",
        "api_parameters",
        "visible_ports",
        "delays",
        "repeat",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PulseDocument has an unknown field set")
    if tree["schema"] != PULSE_DOCUMENT_SCHEMA:
        raise ValueError("PulseDocument schema differs")
    for field in (
        "periods",
        "scan_parameters",
        "api_parameters",
        "visible_ports",
        "delays",
    ):
        if not isinstance(tree[field], list):
            raise TypeError(f"PulseDocument {field} must be a list")
    table = tree["scan_table"]
    recipe = tree["scan_recipe"]
    repeat = tree["repeat"]
    return PulseDocument(
        name=tree["name"],
        target=pulse_target_from_tree(tree["target"]),
        time_step_ns=tree["time_step_ns"],
        periods=tuple(_period_from_tree(item) for item in tree["periods"]),
        scan_parameters=tuple(
            _scan_parameter_from_tree(item) for item in tree["scan_parameters"]
        ),
        scan_table=None if table is None else frozen_scan_table_from_tree(table),
        scan_recipe=None if recipe is None else _scan_recipe_from_tree(recipe),
        api_parameters=tuple(
            _api_parameter_from_tree(item) for item in tree["api_parameters"]
        ),
        visible_ports=tuple(tree["visible_ports"]),
        delays=tuple(_delay_from_tree(item) for item in tree["delays"]),
        repeat=None if repeat is None else _repeat_from_tree(repeat),
    )


def pulse_document_path(path: str | Path) -> Path:
    """Return the sole canonical path used by PulseDocument load and save."""

    destination = Path(path).expanduser().resolve()
    return (
        destination
        if destination.suffix.lower() == ".json"
        else destination.with_suffix(".json")
    )


def save_pulse_document(document: PulseDocument, path: str | Path) -> Path:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    destination = pulse_document_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            pulse_document_to_tree(document),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_pulse_document(path: str | Path) -> PulseDocument:
    source = pulse_document_path(path)
    return pulse_document_from_tree(json.loads(source.read_text(encoding="utf-8")))


def _field_ref_to_tree(value: PulseFieldRef) -> dict[str, object]:
    return {"kind": value.kind, "period_id": value.period_id, "port": value.port}


def _field_ref_from_tree(tree: object) -> PulseFieldRef:
    if not isinstance(tree, dict) or set(tree) != {"kind", "period_id", "port"}:
        raise ValueError("PulseFieldRef has an unknown field set")
    return PulseFieldRef(tree["kind"], tree["period_id"], tree["port"])


def _period_to_tree(value: PulsePeriod) -> dict[str, object]:
    return {
        "period_id": value.period_id,
        "duration": value.duration,
        "unit": value.unit,
        "name": value.name,
        "states": list(value.states),
        "analog_steps": [
            {"port": item.port, "mode": item.mode, "value": item.value}
            for item in value.analog_steps
        ],
    }


def _period_from_tree(tree: object) -> PulsePeriod:
    fields = {"period_id", "duration", "unit", "name", "states", "analog_steps"}
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PulsePeriod has an unknown field set")
    if not isinstance(tree["states"], list) or not isinstance(tree["analog_steps"], list):
        raise TypeError("PulsePeriod states and analog_steps must be lists")
    steps = []
    for item in tree["analog_steps"]:
        if not isinstance(item, dict) or set(item) != {"port", "mode", "value"}:
            raise ValueError("AnalogStep has an unknown field set")
        steps.append(AnalogStep(item["port"], item["mode"], item["value"]))
    return PulsePeriod(
        tree["period_id"],
        tree["duration"],
        tree["unit"],
        tree["name"],
        tuple(tree["states"]),
        tuple(steps),
    )


def _scan_parameter_to_tree(value: ScanParameter) -> dict[str, object]:
    return {
        "parameter_id": value.parameter_id,
        "field": _field_ref_to_tree(value.field),
        "label": value.label,
        "unit": value.unit,
    }


def _scan_parameter_from_tree(tree: object) -> ScanParameter:
    if not isinstance(tree, dict) or set(tree) != {"parameter_id", "field", "label", "unit"}:
        raise ValueError("ScanParameter has an unknown field set")
    return ScanParameter(
        tree["parameter_id"],
        _field_ref_from_tree(tree["field"]),
        tree["label"],
        tree["unit"],
    )


def _api_parameter_to_tree(value: ApiParameter) -> dict[str, object]:
    return {
        "parameter_id": value.parameter_id,
        "field": _field_ref_to_tree(value.field),
        "unit": value.unit,
    }


def _api_parameter_from_tree(tree: object) -> ApiParameter:
    if not isinstance(tree, dict) or set(tree) != {"parameter_id", "field", "unit"}:
        raise ValueError("ApiParameter has an unknown field set")
    return ApiParameter(
        tree["parameter_id"],
        _field_ref_from_tree(tree["field"]),
        tree["unit"],
    )


def _scan_recipe_to_tree(value: ScanRecipeProvenance) -> dict[str, object]:
    return {
        "language": value.language,
        "source": value.source,
        "columns": list(value.columns),
        "normalizer_id": value.normalizer_id,
        "frozen_definition_digest": value.frozen_definition_digest,
    }


def _scan_recipe_from_tree(tree: object) -> ScanRecipeProvenance:
    fields = {
        "language",
        "source",
        "columns",
        "normalizer_id",
        "frozen_definition_digest",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("ScanRecipeProvenance has an unknown field set")
    if not isinstance(tree["columns"], list):
        raise TypeError("ScanRecipeProvenance columns must be a list")
    return ScanRecipeProvenance(
        tree["language"],
        tree["source"],
        tuple(tree["columns"]),
        tree["normalizer_id"],
        tree["frozen_definition_digest"],
    )


def _delay_to_tree(value: OutputDelay) -> dict[str, object]:
    return {"port": value.port, "value": value.value, "unit": value.unit}


def _delay_from_tree(tree: object) -> OutputDelay:
    if not isinstance(tree, dict) or set(tree) != {"port", "value", "unit"}:
        raise ValueError("OutputDelay has an unknown field set")
    return OutputDelay(tree["port"], tree["value"], tree["unit"])


def _repeat_to_tree(value: RepeatRegion) -> dict[str, object]:
    return {
        "start_period_id": value.start_period_id,
        "end_period_id": value.end_period_id,
        "count": value.count,
    }


def _repeat_from_tree(tree: object) -> RepeatRegion:
    if not isinstance(tree, dict) or set(tree) != {
        "start_period_id",
        "end_period_id",
        "count",
    }:
        raise ValueError("RepeatRegion has an unknown field set")
    return RepeatRegion(tree["start_period_id"], tree["end_period_id"], tree["count"])


__all__ = [
    "ANALOG_MODES",
    "ApiParameter",
    "AnalogStep",
    "FIELD_DAC",
    "FIELD_DELAY",
    "FIELD_DURATION",
    "FIELD_KINDS",
    "FrozenScanTable",
    "OutputDelay",
    "PULSE_DOCUMENT_SCHEMA",
    "PulseDocument",
    "PulseFieldRef",
    "PulsePeriod",
    "RepeatRegion",
    "SCAN_FIELD_KINDS",
    "SCAN_NORMALIZER_ID",
    "SCAN_TABLE_SCHEMA",
    "ScanParameter",
    "ScanRecipeProvenance",
    "TIME_UNITS",
    "TIME_UNIT_TO_NS",
    "frozen_scan_table_from_tree",
    "frozen_scan_table_to_tree",
    "load_pulse_document",
    "pulse_document_from_tree",
    "pulse_document_path",
    "pulse_document_to_tree",
    "save_pulse_document",
]
