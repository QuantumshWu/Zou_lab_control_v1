"""Immutable current pulse authoring document and current-only codec."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from zlc_storage import canonical_digest

from .target import PORT_DAC, PulseTarget, pulse_target_from_tree, pulse_target_to_tree


PULSE_DOCUMENT_SCHEMA = "zlc_pulse.PulseDocument/v1"
TIME_UNITS = frozenset(("s", "ms", "us", "ns"))
SCAN_KINDS = frozenset(("duration", "dac"))
API_KINDS = frozenset(("duration", "delay", "dac"))
ANALOG_MODES = frozenset(("hold", "edge", "ramp"))
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_API_NAME = re.compile(r"a[1-9][0-9]*\Z")


def _text(value: object, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value or (not empty and not value):
        qualifier = "canonical text" if empty else "canonical non-empty text"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def _integer(
    value: object,
    field: str,
    *,
    optional: bool = False,
    nonnegative: bool = False,
) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        suffix = " or None" if optional else ""
        raise TypeError(f"{field} must be an integer{suffix}")
    if nonnegative and value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _number(value: object, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    return value


def _expression(value: object, field: str) -> int | float | str:
    if isinstance(value, str):
        return _text(value, field)
    return _number(value, field)


@dataclass(frozen=True)
class PulsePeriod:
    duration: int | float | str
    unit: str
    name: str
    states: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration", _expression(self.duration, "period duration"))
        unit = _text(self.unit, "period unit")
        if unit not in TIME_UNITS:
            raise ValueError(f"unsupported period unit {unit!r}")
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "name", _text(self.name, "period name", empty=True))
        states = tuple(self.states)
        if any(type(value) not in (bool, int) or int(value) not in (0, 1) for value in states):
            raise ValueError("period states must contain only boolean/0/1 values")
        object.__setattr__(self, "states", tuple(int(bool(value)) for value in states))


@dataclass(frozen=True)
class ScanSlot:
    name: str
    kind: str
    target: str
    label: str
    unit: str
    nominal: int | float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "scan slot name"))
        if _IDENTIFIER.fullmatch(self.name) is None:
            raise ValueError("scan slot name must be an identifier")
        kind = _text(self.kind, "scan slot kind")
        if kind not in SCAN_KINDS:
            raise ValueError(f"unsupported scan slot kind {kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target", _text(self.target, "scan slot target"))
        object.__setattr__(self, "label", _text(self.label, "scan slot label", empty=True))
        unit = _text(self.unit, "scan slot unit")
        expected = "value" if kind == "dac" else None
        if (expected is not None and unit != expected) or (
            expected is None and unit not in TIME_UNITS
        ):
            raise ValueError("scan slot unit is incompatible with its kind")
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "nominal", _number(self.nominal, "scan slot nominal"))


@dataclass(frozen=True)
class ApiSlot:
    name: str
    kind: str
    target: str
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "API slot name"))
        if _API_NAME.fullmatch(self.name) is None:
            raise ValueError("API slot name must look like a1/a2")
        kind = _text(self.kind, "API slot kind")
        if kind not in API_KINDS:
            raise ValueError(f"unsupported API slot kind {kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target", _text(self.target, "API slot target"))
        unit = _text(self.unit, "API slot unit")
        if kind == "dac":
            unit = "value"
        elif unit not in TIME_UNITS:
            raise ValueError("time API slots require a supported time unit")
        object.__setattr__(self, "unit", unit)


@dataclass(frozen=True)
class AnalogBusStep:
    mode: str
    value: int | float | str | None

    def __post_init__(self) -> None:
        mode = _text(self.mode, "analog bus mode")
        if mode not in ANALOG_MODES:
            raise ValueError(f"unsupported analog bus mode {mode!r}")
        if mode == "hold" and self.value is not None:
            raise ValueError("hold analog bus steps cannot carry a value")
        if mode != "hold" and self.value is None:
            raise ValueError(f"{mode} analog bus steps require a value")
        object.__setattr__(self, "mode", mode)
        if self.value is not None:
            object.__setattr__(self, "value", _expression(self.value, "analog bus value"))


@dataclass(frozen=True)
class PulseDocument:
    name: str
    target: PulseTarget
    time_step_ns: float
    periods: tuple[PulsePeriod, ...]
    scan_slots: tuple[ScanSlot, ...] = ()
    scan_table: tuple[tuple[float, ...], ...] = ()
    scan_code: str = ""
    api_slots: tuple[ApiSlot, ...] = ()
    visible_ports: tuple[str, ...] = ()
    analog_bus_programs: tuple[tuple[str, tuple[AnalogBusStep, ...]], ...] = ()
    delays: tuple[tuple[str, int | float | str], ...] = ()
    delay_units: tuple[tuple[str, str], ...] = ()
    repeat_start: int | None = None
    repeat_end: int | None = None
    repeat_count: int = 1
    repeat_forever: bool = False
    scan_repeats: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "pulse document name"))
        if not isinstance(self.target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        step = float(_number(self.time_step_ns, "time_step_ns"))
        if step <= 0:
            raise ValueError("time_step_ns must be positive")
        object.__setattr__(self, "time_step_ns", step)
        periods = tuple(self.periods)
        if not periods or any(not isinstance(period, PulsePeriod) for period in periods):
            raise ValueError("periods must contain PulsePeriod values")
        if any(len(period.states) != len(self.target.raw_lanes) for period in periods):
            raise ValueError("every period state vector must match target raw lanes")
        object.__setattr__(self, "periods", periods)
        slots = tuple(self.scan_slots)
        if any(not isinstance(slot, ScanSlot) for slot in slots):
            raise TypeError("scan_slots must contain ScanSlot values")
        if len({slot.name for slot in slots}) != len(slots):
            raise ValueError("scan slot names must be unique")
        object.__setattr__(self, "scan_slots", slots)
        rows = tuple(
            tuple(float(_number(value, "scan table value")) for value in row)
            for row in self.scan_table
        )
        if any(len(row) != len(slots) for row in rows):
            raise ValueError("every scan table row must match scan slot cardinality")
        if rows and not slots:
            raise ValueError("a scan table requires bound scan slots")
        object.__setattr__(self, "scan_table", rows)
        if not isinstance(self.scan_code, str):
            raise TypeError("scan_code must be text")
        api = tuple(self.api_slots)
        if any(not isinstance(slot, ApiSlot) for slot in api):
            raise TypeError("api_slots must contain ApiSlot values")
        if len({slot.name for slot in api}) != len(api):
            raise ValueError("API slot names must be unique")
        object.__setattr__(self, "api_slots", api)
        visible = tuple(_text(key, "visible port") for key in self.visible_ports)
        if len(set(visible)) != len(visible) or any(
            key not in self.target.by_key for key in visible
        ):
            raise ValueError("visible_ports must be unique keys in the PulseTarget")
        object.__setattr__(self, "visible_ports", visible)
        programs = tuple(
            (_text(key, "analog bus program key"), tuple(steps))
            for key, steps in self.analog_bus_programs
        )
        if len({key for key, _steps in programs}) != len(programs):
            raise ValueError("analog bus program keys must be unique")
        for key, steps in programs:
            port = self.target.by_key.get(key)
            if port is None or port.kind != PORT_DAC:
                raise ValueError(f"analog bus program {key!r} is not a target DAC")
            if len(steps) != len(periods) or any(
                not isinstance(item, AnalogBusStep) for item in steps
            ):
                raise ValueError("each analog bus program must define one step per period")
        object.__setattr__(self, "analog_bus_programs", programs)
        delays = tuple(
            (_text(key, "delay lane"), _expression(value, "delay value"))
            for key, value in self.delays
        )
        units = tuple(
            (_text(key, "delay unit lane"), _text(unit, "delay unit"))
            for key, unit in self.delay_units
        )
        delay_keys = tuple(key for key, _value in delays)
        unit_keys = tuple(key for key, _unit in units)
        if (
            len(set(delay_keys)) != len(delays)
            or len(set(unit_keys)) != len(units)
            or set(delay_keys) != set(unit_keys)
        ):
            raise ValueError("delays and delay_units must have the same unique lane keys")
        if any(key not in self.target.raw_lanes for key in delay_keys):
            raise ValueError("delay keys must name target raw lanes")
        if any(unit not in TIME_UNITS for _key, unit in units):
            raise ValueError("delay unit is unsupported")
        object.__setattr__(self, "delays", delays)
        object.__setattr__(self, "delay_units", units)
        start = _integer(self.repeat_start, "repeat_start", optional=True, nonnegative=True)
        end = _integer(self.repeat_end, "repeat_end", optional=True, nonnegative=True)
        if (start is None) != (end is None):
            raise ValueError("repeat_start and repeat_end must be set together")
        if start is not None and (end < start or end >= len(periods)):
            raise ValueError("repeat bracket is outside the period table")
        count = _integer(self.repeat_count, "repeat_count", nonnegative=True)
        assert count is not None
        if count < 1:
            raise ValueError("repeat_count must be at least one")
        if type(self.repeat_forever) is not bool:
            raise TypeError("repeat_forever must be bool")
        scan_repeats = _integer(self.scan_repeats, "scan_repeats", nonnegative=True)
        assert scan_repeats is not None
        object.__setattr__(self, "repeat_start", start)
        object.__setattr__(self, "repeat_end", end)
        object.__setattr__(self, "repeat_count", count)
        object.__setattr__(self, "scan_repeats", scan_repeats)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(pulse_document_to_tree(self))

    def save(self, path: str | Path) -> Path:
        return save_pulse_document(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PulseDocument":
        from .legacy import load_pulse_document

        return load_pulse_document(path)


def pulse_document_to_tree(document: PulseDocument) -> dict[str, object]:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    return {
        "schema": PULSE_DOCUMENT_SCHEMA,
        "name": document.name,
        "target": pulse_target_to_tree(document.target),
        "time_step_ns": document.time_step_ns,
        "periods": [_period_to_tree(period) for period in document.periods],
        "scan_slots": [_scan_slot_to_tree(slot) for slot in document.scan_slots],
        "scan_table": [list(row) for row in document.scan_table],
        "scan_code": document.scan_code,
        "api_slots": [_api_slot_to_tree(slot) for slot in document.api_slots],
        "visible_ports": list(document.visible_ports),
        "analog_bus_programs": [
            {
                "port": key,
                "steps": [
                    {"mode": step.mode, "value": step.value} for step in steps
                ],
            }
            for key, steps in document.analog_bus_programs
        ],
        "delays": [
            {"lane": key, "value": value, "unit": dict(document.delay_units)[key]}
            for key, value in document.delays
        ],
        "repeat": {
            "start": document.repeat_start,
            "end": document.repeat_end,
            "count": document.repeat_count,
            "forever": document.repeat_forever,
        },
        "scan_repeats": document.scan_repeats,
    }


def pulse_document_from_tree(tree: object) -> PulseDocument:
    fields = {
        "schema",
        "name",
        "target",
        "time_step_ns",
        "periods",
        "scan_slots",
        "scan_table",
        "scan_code",
        "api_slots",
        "visible_ports",
        "analog_bus_programs",
        "delays",
        "repeat",
        "scan_repeats",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PulseDocument has an unknown field set")
    if tree["schema"] != PULSE_DOCUMENT_SCHEMA:
        raise ValueError("PulseDocument schema differs")
    for field in (
        "periods",
        "scan_slots",
        "scan_table",
        "api_slots",
        "visible_ports",
        "analog_bus_programs",
        "delays",
    ):
        if not isinstance(tree[field], list):
            raise TypeError(f"PulseDocument {field} must be a list")
    repeat = tree["repeat"]
    if not isinstance(repeat, dict) or set(repeat) != {"start", "end", "count", "forever"}:
        raise ValueError("PulseDocument repeat has an unknown field set")
    programs = []
    for item in tree["analog_bus_programs"]:
        if not isinstance(item, dict) or set(item) != {"port", "steps"}:
            raise ValueError("analog bus program has an unknown field set")
        if not isinstance(item["steps"], list):
            raise TypeError("analog bus steps must be a list")
        steps = []
        for step in item["steps"]:
            if not isinstance(step, dict) or set(step) != {"mode", "value"}:
                raise ValueError("analog bus step has an unknown field set")
            steps.append(AnalogBusStep(step["mode"], step["value"]))
        programs.append((item["port"], tuple(steps)))
    delays = []
    units = []
    for item in tree["delays"]:
        if not isinstance(item, dict) or set(item) != {"lane", "value", "unit"}:
            raise ValueError("delay has an unknown field set")
        delays.append((item["lane"], item["value"]))
        units.append((item["lane"], item["unit"]))
    return PulseDocument(
        tree["name"],
        pulse_target_from_tree(tree["target"]),
        tree["time_step_ns"],
        tuple(_period_from_tree(item) for item in tree["periods"]),
        tuple(_scan_slot_from_tree(item) for item in tree["scan_slots"]),
        tuple(_scan_row(row) for row in tree["scan_table"]),
        tree["scan_code"],
        tuple(_api_slot_from_tree(item) for item in tree["api_slots"]),
        tuple(tree["visible_ports"]),
        tuple(programs),
        tuple(delays),
        tuple(units),
        repeat["start"],
        repeat["end"],
        repeat["count"],
        repeat["forever"],
        tree["scan_repeats"],
    )


def save_pulse_document(document: PulseDocument, path: str | Path) -> Path:
    if not isinstance(document, PulseDocument):
        raise TypeError("save requires PulseDocument")
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        pulse_document_to_tree(document),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    destination.write_text(payload, encoding="utf-8")
    return destination


def _period_to_tree(value: PulsePeriod) -> dict[str, object]:
    return {
        "duration": value.duration,
        "unit": value.unit,
        "name": value.name,
        "states": list(value.states),
    }


def _period_from_tree(tree: object) -> PulsePeriod:
    if not isinstance(tree, dict) or set(tree) != {"duration", "unit", "name", "states"}:
        raise ValueError("pulse period has an unknown field set")
    if not isinstance(tree["states"], list):
        raise TypeError("pulse period states must be a list")
    return PulsePeriod(tree["duration"], tree["unit"], tree["name"], tuple(tree["states"]))


def _scan_slot_to_tree(value: ScanSlot) -> dict[str, object]:
    return {
        "name": value.name,
        "kind": value.kind,
        "target": value.target,
        "label": value.label,
        "unit": value.unit,
        "nominal": value.nominal,
    }


def _scan_slot_from_tree(tree: object) -> ScanSlot:
    if not isinstance(tree, dict) or set(tree) != {
        "name",
        "kind",
        "target",
        "label",
        "unit",
        "nominal",
    }:
        raise ValueError("scan slot has an unknown field set")
    return ScanSlot(**tree)


def _api_slot_to_tree(value: ApiSlot) -> dict[str, object]:
    return {"name": value.name, "kind": value.kind, "target": value.target, "unit": value.unit}


def _api_slot_from_tree(tree: object) -> ApiSlot:
    if not isinstance(tree, dict) or set(tree) != {"name", "kind", "target", "unit"}:
        raise ValueError("API slot has an unknown field set")
    return ApiSlot(**tree)


def _scan_row(row: object) -> tuple[object, ...]:
    if not isinstance(row, list):
        raise TypeError("scan table rows must be lists")
    return tuple(row)


__all__ = [
    "ANALOG_MODES",
    "API_KINDS",
    "ApiSlot",
    "AnalogBusStep",
    "PULSE_DOCUMENT_SCHEMA",
    "PulseDocument",
    "PulsePeriod",
    "SCAN_KINDS",
    "ScanSlot",
    "TIME_UNITS",
    "pulse_document_from_tree",
    "pulse_document_to_tree",
    "save_pulse_document",
]
