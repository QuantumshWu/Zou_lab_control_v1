"""The sole historical pulse JSON loader.

Current in-memory construction and saving never call this module.  It accepts
the historical PulseTableState/PulseSequence file schemas, translates once to
PulseDocument, and discards the historical vocabulary.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .document import (
    PULSE_DOCUMENT_SCHEMA,
    AnalogBusStep,
    ApiSlot,
    PulseDocument,
    PulsePeriod,
    ScanSlot,
    pulse_document_from_tree,
)
from .target import (
    PulseTarget,
    pulse_target_from_legacy_channels,
    pulse_target_from_legacy_tree,
)


_OLD_TABLE_SCHEMA = "Zou_lab_control.neutral_atom.PulseTableState"
_OLD_SEQUENCE_SCHEMA = "Zou_lab_control.neutral_atom.PulseSequence"
_SCAN_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _legacy_unit(value: object, *, dac: bool = False) -> str:
    if dac:
        return "value"
    text = str(value or "ns")
    return "ns" if text == "str (ns)" else text


def load_pulse_document(path: str | Path) -> PulseDocument:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pulse JSON in {source} must contain one object")
    schema = payload.get("schema")
    if schema == PULSE_DOCUMENT_SCHEMA:
        return pulse_document_from_tree(payload)
    if schema == _OLD_TABLE_SCHEMA:
        return pulse_document_from_legacy_table(payload)
    if schema == _OLD_SEQUENCE_SCHEMA:
        return pulse_document_from_legacy_sequence(payload)
    raise ValueError(f"unsupported pulse JSON schema in {source}: {schema!r}")


def pulse_document_from_legacy_table(payload: object) -> PulseDocument:
    if not isinstance(payload, dict):
        raise TypeError("legacy pulse table must be an object")
    if payload.get("schema") != _OLD_TABLE_SCHEMA:
        raise ValueError("legacy pulse table schema differs")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1 or version > 4:
        raise ValueError(f"unsupported legacy pulse table version {version!r}")
    upgraded = _upgrade_table(payload, version)
    target = upgraded["target"]
    assert isinstance(target, PulseTarget)
    periods = tuple(_legacy_period(item) for item in upgraded["periods"])
    slots = tuple(_legacy_scan_slot(item) for item in upgraded["scan_slots"])
    api = tuple(_legacy_api_slot(item) for item in upgraded["api_slots"])
    delays = tuple((str(key), value) for key, value in upgraded["delays"].items())
    delay_units = tuple(
        (key, _legacy_unit(upgraded["delay_units"].get(key, "ns")))
        for key, _value in delays
    )
    programs = []
    for key, steps in upgraded["analog_bus_modes"].items():
        if not isinstance(steps, list):
            raise TypeError("legacy analog bus program must be a list")
        converted = []
        for step in steps:
            if not isinstance(step, dict):
                raise TypeError("legacy analog bus step must be an object")
            allowed = {"mode", "value"}
            if set(step) != allowed:
                raise ValueError("legacy analog bus step has an unknown field set")
            converted.append(AnalogBusStep(str(step["mode"]), step["value"]))
        programs.append((str(key), tuple(converted)))
    return PulseDocument(
        str(upgraded["name"]),
        target,
        float(upgraded["time_step_ns"]),
        periods,
        slots,
        tuple(tuple(float(value) for value in row) for row in upgraded["scan_table"]),
        str(upgraded["scan_code"]),
        api,
        tuple(str(key) for key in upgraded["visible_ports"]),
        tuple(programs),
        delays,
        delay_units,
        upgraded["repeat_start"],
        upgraded["repeat_end"],
        int(upgraded["repeat_count"]),
        bool(upgraded["repeat_forever"]),
        int(upgraded["scan_repeats"]),
    )


def pulse_document_from_legacy_sequence(payload: object) -> PulseDocument:
    if not isinstance(payload, dict):
        raise TypeError("legacy PulseSequence must be an object")
    if payload.get("schema") != _OLD_SEQUENCE_SCHEMA or payload.get("version") != 2:
        raise ValueError("unsupported legacy PulseSequence schema/version")
    expected = {
        "schema",
        "version",
        "name",
        "delays",
        "repeat_count",
        "repeat_period",
        "repeat_forever",
        "pulses",
        "source_table",
    }
    if set(payload) != expected:
        raise ValueError("legacy PulseSequence has an unknown field set")
    source_table = payload["source_table"]
    if isinstance(source_table, dict):
        return pulse_document_from_legacy_table(source_table)
    pulses = payload["pulses"]
    delays = payload["delays"]
    if not isinstance(pulses, list) or not isinstance(delays, dict):
        raise TypeError("legacy PulseSequence pulses/delays are invalid")
    parsed = [_legacy_flat_pulse(item) for item in pulses]
    repeat_count = payload["repeat_count"]
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
        raise TypeError("legacy repeat_count must be an integer")
    if type(payload["repeat_forever"]) is not bool:
        raise TypeError("legacy repeat_forever must be bool")
    lanes: list[str] = []
    for pulse in parsed:
        if pulse["channel"] not in lanes:
            lanes.append(pulse["channel"])
    for lane in delays:
        if str(lane) not in lanes:
            lanes.append(str(lane))
    if not lanes:
        raise ValueError("legacy PulseSequence has no output channels")
    target = pulse_target_from_legacy_channels(lanes)
    end = max((pulse["stop"] for pulse in parsed), default=0.0)
    repeat_period = payload["repeat_period"]
    if repeat_period is not None:
        if isinstance(repeat_period, bool) or not isinstance(repeat_period, (int, float)):
            raise TypeError("legacy repeat_period must be numeric or null")
        if not math.isfinite(float(repeat_period)) or float(repeat_period) < end:
            raise ValueError("legacy repeat_period is shorter than its pulses")
        end = float(repeat_period)
    boundaries = sorted({0.0, end, *(pulse["start"] for pulse in parsed), *(pulse["stop"] for pulse in parsed)})
    periods = []
    for index, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        if stop <= start:
            continue
        midpoint = (start + stop) / 2.0
        states = []
        active_names = []
        for lane in lanes:
            active = [
                pulse
                for pulse in parsed
                if pulse["channel"] == lane and pulse["start"] <= midpoint < pulse["stop"]
            ]
            if len(active) > 1:
                raise ValueError(
                    f"legacy PulseSequence has overlapping pulses on {lane!r}"
                )
            states.append(active[-1]["value"] if active else 0)
            active_names.extend(pulse["name"] for pulse in active if pulse["name"])
        name = "+".join(dict.fromkeys(active_names)) or f"period_{index}"
        periods.append(PulsePeriod(stop - start, "s", name, tuple(states)))
    if not periods:
        raise ValueError("legacy PulseSequence has zero duration")
    delay_pairs = tuple((str(key), value) for key, value in delays.items())
    return PulseDocument(
        str(payload["name"]),
        target,
        1.0,
        tuple(periods),
        visible_ports=tuple(port.key for port in target.ports),
        delays=delay_pairs,
        delay_units=tuple((key, "s") for key, _value in delay_pairs),
        repeat_start=0,
        repeat_end=len(periods) - 1,
        repeat_count=repeat_count,
        repeat_forever=payload["repeat_forever"],
    )


def _upgrade_table(payload: dict, version: int) -> dict:
    if version == 4:
        fields = {
            "schema",
            "version",
            "name",
            "port_catalog",
            "scan_slots",
            "scan_table",
            "scan_code",
            "api_slots",
            "time_step_ns",
            "periods",
            "visible_ports",
            "analog_bus_modes",
            "delays",
            "delay_units",
            "repeat_start",
            "repeat_end",
            "repeat_count",
            "repeat_forever",
            "scan_repeats",
        }
        if set(payload) != fields:
            raise ValueError("legacy v4 pulse table has an unknown field set")
        target = pulse_target_from_legacy_tree(payload["port_catalog"])
        visible = list(payload["visible_ports"])
    else:
        target = pulse_target_from_legacy_channels(
            payload.get("channels") or (),
            channel_labels=payload.get("channel_labels") or {},
            analog_buses=payload.get("analog_buses") or {},
            clk_channels=payload.get("clk_channels") or (),
        )
        visible = []
        for lane in payload.get("visible_channels") or ():
            lane = str(lane)
            for port in target.ports:
                if lane in port.lanes and port.key not in visible:
                    visible.append(port.key)
                    break
        if not visible:
            visible = [port.key for port in target.ports]
    periods = payload.get("periods") or ()
    if not isinstance(periods, (list, tuple)):
        raise TypeError("legacy periods must be an ordered array")
    return {
        "name": str(payload.get("name") or "pulse"),
        "target": target,
        "scan_slots": list(payload.get("scan_slots") or ()),
        "scan_table": list(payload.get("scan_table") or ()),
        "scan_code": str(payload.get("scan_code") or ""),
        "api_slots": list(payload.get("api_slots") or ()),
        "time_step_ns": float(payload.get("time_step_ns") or 1.0),
        "periods": list(periods),
        "visible_ports": visible,
        "analog_bus_modes": dict(payload.get("analog_bus_modes") or {}),
        "delays": dict(payload.get("delays") or {}),
        "delay_units": dict(payload.get("delay_units") or {}),
        "repeat_start": payload.get("repeat_start"),
        "repeat_end": payload.get("repeat_end"),
        "repeat_count": 1 if payload.get("repeat_count") is None else int(payload["repeat_count"]),
        "repeat_forever": bool(payload.get("repeat_forever", True)),
        "scan_repeats": 0 if payload.get("scan_repeats") is None else int(payload["scan_repeats"]),
    }


def _legacy_period(tree: object) -> PulsePeriod:
    if not isinstance(tree, dict):
        raise TypeError("legacy period must be an object")
    if not {"duration", "states"}.issubset(tree) or set(tree) - {
        "duration",
        "states",
        "unit",
        "name",
    }:
        raise ValueError("legacy period has an unknown field set")
    states = tree["states"]
    if not isinstance(states, (list, tuple)):
        raise TypeError("legacy period states must be an ordered array")
    return PulsePeriod(
        tree["duration"],
        _legacy_unit(tree.get("unit")),
        str(tree.get("name") or ""),
        tuple(states),
    )


def _legacy_scan_slot(tree: object) -> ScanSlot:
    if not isinstance(tree, dict):
        raise TypeError("legacy scan slot must be an object")
    if not {"kind", "target"}.issubset(tree) or set(tree) - {
        "name",
        "kind",
        "target",
        "label",
        "unit",
        "nominal",
    }:
        raise ValueError("legacy scan slot has an unknown field set")
    label = str(tree.get("label") or "")
    name = str(tree.get("name") or "").strip()
    if not name:
        name = re.sub(r"[^0-9A-Za-z_]+", "_", label).strip("_")
        if name and name[0].isdigit():
            name = "scan_" + name
        if not name:
            name = f"scan_{tree['kind']}_{str(tree['target']).replace('@', '_')}"
    if _SCAN_IDENTIFIER.fullmatch(name) is None:
        raise ValueError("legacy scan slot cannot derive a canonical identifier")
    kind = str(tree["kind"])
    return ScanSlot(
        name,
        kind,
        str(tree["target"]),
        label,
        _legacy_unit(tree.get("unit"), dac=kind == "dac"),
        float(tree.get("nominal") or 0.0),
    )


def _legacy_api_slot(tree: object) -> ApiSlot:
    if not isinstance(tree, dict) or not {"name", "kind", "target"}.issubset(tree) or set(tree) - {
        "name",
        "kind",
        "target",
        "unit",
    }:
        raise ValueError("legacy API slot has an unknown field set")
    return ApiSlot(
        str(tree["name"]),
        str(tree["kind"]),
        str(tree["target"]),
        _legacy_unit(tree.get("unit"), dac=str(tree["kind"]) == "dac"),
    )


def _legacy_flat_pulse(tree: object) -> dict[str, object]:
    if not isinstance(tree, dict) or set(tree) != {
        "channel",
        "start",
        "duration",
        "value",
        "name",
    }:
        raise ValueError("legacy flat pulse has an unknown field set")
    start = float(tree["start"])
    duration = float(tree["duration"])
    if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
        raise ValueError("legacy flat pulse timing is invalid")
    value = tree["value"]
    if type(value) not in (bool, int) or int(value) not in (0, 1):
        raise ValueError("legacy flat pulse value must be digital")
    return {
        "channel": str(tree["channel"]),
        "start": start,
        "stop": start + duration,
        "value": int(bool(value)),
        "name": str(tree["name"]),
    }


__all__ = [
    "load_pulse_document",
    "pulse_document_from_legacy_sequence",
    "pulse_document_from_legacy_table",
]
