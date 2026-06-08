"""Confocal-style pulse-table editor model for ``PulseSequence``.

The model keeps the GUI's user-facing idea: a horizontal list of *periods*,
where each period has one duration and a full digital state vector.  It does
not own hardware.  It only compiles to ``PulseSequence`` so notebooks, PyQt,
and remote FPGA sequencers share the same timing source of truth.

Scanning
--------
Any per-field value (a period duration, a channel delay, or an analog-bus DAC
value) can be *bound to a scan slot*.  Slots are named ``s0, s1, ...`` in bind
order.  A bound field's value is taken, per scan point, from one column of a
``scan_table`` (an ``N_points x N_slots`` array, typically loaded from a file).
The hardware iterates the scan-point rows seamlessly; the host only uploads the
sequence template plus the parameter table.  There is exactly one scan concept
(named slots); there is no separate ``x``/``y`` notion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import ast
import json
import math
from numbers import Number
import re
from typing import Iterable, Mapping, Sequence

from .sequence import DEFAULT_CAMERA_TRIGGER_CHANNELS, PulseSequence, channel_names, positive_float


UNITS_TO_NS = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0, "str (ns)": 1.0}
GRID_RTOL = 1e-12
GRID_ATOL_STEPS = 1e-9
BUS_LABEL_RE = re.compile(r"^(?P<base>.+)\[(?P<bit>\d+)\]$")
ANALOG_BUS_MODES = ("hold", "edge", "ramp")

#: Scan-slot kinds.  ``duration`` binds a period duration, ``delay`` binds a
#: channel delay, ``dac`` binds one analog-bus value in one period.
SCAN_SLOT_KINDS = ("duration", "delay", "dac")
SLOT_VAR_RE = re.compile(r"^s(?P<index>\d+)$")


def _cyclic_shift_interval(start: int, stop: int, delay: int, total: int) -> list[tuple[int, int]]:
    """Shift an ON interval ``[start, stop)`` later by ``delay`` steps, CYCLICALLY
    within a frame of ``total`` steps: any piece pushed past ``total`` wraps to the
    front.  Returns 1 or 2 sub-intervals, all inside ``[0, total)``.  This is the
    periodic ("inf") delay the preview always shows, and what the hardware applies for
    a repeat_forever sequence.  Matches Confocal-GUIv2 ``base.delay`` (delay %% total,
    cyclic roll).  Python ``%`` keeps the result in ``[0, total)`` so a negative delay
    wraps correctly too."""
    if total <= 0:
        return [(start, stop)]
    d = delay % total
    a, b = start + d, stop + d
    if b <= total:
        return [(a, b)]
    if a >= total:
        return [(a - total, b - total)]
    return [(a, total), (0, b - total)]


def default_pulse_name() -> str:
    return "pulse_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def slot_var(index: int) -> str:
    """Return the expression variable name for scan slot ``index`` (``s0``...)."""

    if int(index) < 0:
        raise ValueError("scan slot index must be >= 0.")
    return f"s{int(index)}"


@dataclass(frozen=True)
class ScanSlot:
    """One bound scan parameter.

    ``kind`` is one of :data:`SCAN_SLOT_KINDS`.  ``target`` identifies the bound
    field: a period index (``duration``), a channel name (``delay``), or
    ``"<bus>@<period_index>"`` (``dac``).  ``label`` is a short human name for
    GUI lists.  ``unit`` records the field's display unit; ``scan_table`` values
    are stored as the field's final physical quantity (ns for time slots, the
    integer DAC code for ``dac`` slots).
    """

    kind: str
    target: str
    label: str = ""
    unit: str = "ns"
    nominal: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in SCAN_SLOT_KINDS:
            raise ValueError(f"scan slot kind must be one of {SCAN_SLOT_KINDS}, got {self.kind!r}.")
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "unit", str(self.unit))
        object.__setattr__(self, "nominal", float(self.nominal))

    @property
    def is_time(self) -> bool:
        return self.kind in ("duration", "delay")

    @property
    def dac_bus(self) -> str:
        if self.kind != "dac":
            raise ValueError("dac_bus is only defined for dac slots.")
        return self.target.split("@", 1)[0]

    @property
    def dac_period(self) -> int:
        if self.kind != "dac":
            raise ValueError("dac_period is only defined for dac slots.")
        return int(self.target.split("@", 1)[1])

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "target": self.target, "label": self.label, "unit": self.unit, "nominal": self.nominal}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ScanSlot":
        return cls(
            kind=str(payload.get("kind", "duration")),
            target=str(payload.get("target", "")),
            label=str(payload.get("label", "")),
            unit=str(payload.get("unit", "ns")),
            nominal=float(payload.get("nominal", 0.0)),
        )


@dataclass(frozen=True)
class PulsePeriod:
    """One period-card in the pulse GUI."""

    duration: float | str
    states: tuple[int, ...]
    unit: str = "ns"
    name: str = ""

    def duration_steps(self, *, slots: Mapping[str, float] | None = None, time_step_ns: float = 1.0) -> int:
        value = eval_time_expr(self.duration, slots=slots)
        unit = str(self.unit or "ns")
        if unit not in UNITS_TO_NS:
            raise ValueError(f"unsupported pulse duration unit {unit!r}.")
        return quantized_time_steps(value * UNITS_TO_NS[unit], time_step_ns=time_step_ns, name="period duration", allow_zero=False)

    def duration_ns(self, *, slots: Mapping[str, float] | None = None, time_step_ns: float | None = None) -> float:
        value = eval_time_expr(self.duration, slots=slots)
        unit = str(self.unit or "ns")
        if unit not in UNITS_TO_NS:
            raise ValueError(f"unsupported pulse duration unit {unit!r}.")
        out = value * UNITS_TO_NS[unit]
        if time_step_ns is not None:
            return quantized_time_ns(out, time_step_ns=time_step_ns, name="period duration", allow_zero=False)
        if out <= 0:
            raise ValueError("period duration must be > 0 ns.")
        return out

    def to_dict(self) -> dict[str, object]:
        return {"duration": self.duration, "unit": self.unit, "name": self.name, "states": list(self.states)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PulsePeriod":
        return cls(
            duration=payload.get("duration", 10),
            unit=str(payload.get("unit", "ns")),
            name=str(payload.get("name", "")),
            states=tuple(int(bool(v)) for v in payload.get("states", ())),
        )


class PulseTableState:
    """Editable period table that compiles into a ``PulseSequence``."""

    schema = "Zou_lab_control.neutral_atom.PulseTableState"
    version = 3

    def __init__(
        self,
        *,
        channels: Sequence[str],
        periods: Iterable[PulsePeriod] | None = None,
        delays: Mapping[str, float | str] | None = None,
        delay_units: Mapping[str, str] | None = None,
        name: str | None = None,
        scan_slots: Sequence[ScanSlot | Mapping[str, object]] | None = None,
        scan_table: Sequence[Sequence[float]] | None = None,
        time_step_ns: float = 1.0,
        repeat_start: int | None = None,
        repeat_end: int | None = None,
        repeat_count: int = 1,
        repeat_forever: bool = True,
        visible_channels: Sequence[str] | None = None,
        channel_labels: Mapping[str, str] | None = None,
        analog_buses: Mapping[str, Sequence[str]] | None = None,
        analog_bus_modes: Mapping[str, Sequence[Mapping[str, object] | str | None]] | None = None,
    ):
        self.channels = list(channel_names(channels, "channels"))
        self.name = str(name) if name is not None else default_pulse_name()
        self.time_step_ns = positive_time_step_ns(time_step_ns)
        self.scan_slots = [slot if isinstance(slot, ScanSlot) else ScanSlot.from_dict(slot) for slot in (scan_slots or [])]
        self.scan_table = _normalize_scan_table(scan_table, n_slots=len(self.scan_slots))
        self.periods = list(periods or default_periods(self.channels))
        self.delays = {str(k): v for k, v in dict(delays or {}).items()}
        self.delay_units = {str(k): str(v) for k, v in dict(delay_units or {}).items()}
        self.repeat_start = None if repeat_start is None else int(repeat_start)
        self.repeat_end = None if repeat_end is None else int(repeat_end)
        self.repeat_count = int(repeat_count)
        self.repeat_forever = bool(repeat_forever)
        self.visible_channels = list(channel_names(visible_channels, "visible_channels", allow_empty=True)) if visible_channels is not None else default_visible_channels(self.channels)
        self.channel_labels = {str(k): str(v) for k, v in dict(channel_labels or {}).items()}
        self.analog_buses = {
            str(name): list(channel_names(members, f"analog bus {name!r}"))
            for name, members in dict(analog_buses or {}).items()
        }
        self.analog_bus_modes = self._normalize_analog_bus_modes(analog_bus_modes)
        self.validate()

    # -- scan slot helpers -------------------------------------------------

    @property
    def slot_count(self) -> int:
        return len(self.scan_slots)

    @property
    def scan_var_names(self) -> list[str]:
        return [slot_var(index) for index in range(len(self.scan_slots))]

    @property
    def scan_enabled(self) -> bool:
        return bool(self.scan_slots)

    @property
    def n_points(self) -> int:
        return len(self.scan_table)

    def slot_point(self, point_index: int) -> dict[str, float]:
        """Return ``{s0: value, ...}`` for one scan-table row (native units)."""

        row = self.scan_table[int(point_index)]
        return {slot_var(index): float(row[index]) for index in range(len(self.scan_slots))}

    def slot_point_ns(self, point_index: int) -> dict[str, float]:
        """Return slot values converted to ns for time slots (dac slots pass through)."""

        row = self.scan_table[int(point_index)]
        out: dict[str, float] = {}
        for index, slot in enumerate(self.scan_slots):
            value = float(row[index])
            if slot.is_time:
                value *= UNITS_TO_NS.get(slot.unit, 1.0)
            out[slot_var(index)] = value
        return out

    def reference_slots(self) -> dict[str, float]:
        """Slot values for previewing/validating a non-scan render.

        Uses the first scan point if a table exists, else each slot's nominal
        (the field's value when it was bound).  Time slots are returned in ns.
        """

        if self.scan_table:
            return self.slot_point_ns(0)
        out: dict[str, float] = {}
        for index, slot in enumerate(self.scan_slots):
            value = float(slot.nominal)
            if slot.is_time:
                value *= UNITS_TO_NS.get(slot.unit, 1.0)
            out[slot_var(index)] = value
        return out

    def _read_field_nominal(self, kind: str, target: str, unit: str) -> float:
        """Read a field's current value (in ``unit``) before it is bound."""

        scale = UNITS_TO_NS.get(unit, 1.0)
        try:
            slots = self.reference_slots()
            if kind == "duration":
                return self.periods[int(target)].duration_ns(slots=slots) / scale
            if kind == "delay":
                return self.delay_ns(target, slots=slots) / scale
            if kind == "dac":
                bus, period_index = target.split("@", 1)
                return float(self.bus_value(int(period_index), bus))
        except Exception:
            pass
        return (1000.0 / scale) if kind == "duration" else 0.0

    def bind_field(self, kind: str, target: str, *, label: str = "", unit: str = "ns", nominal: float | None = None) -> int:
        """Bind a field to a new scan slot and rewrite the field to ``s{i}``.

        Returns the new slot index.  Idempotent: re-binding an already bound
        field returns its existing slot index.
        """

        existing = self.slot_index_for(kind, target)
        if existing is not None:
            return existing
        if nominal is None:
            nominal = self._read_field_nominal(kind, str(target), unit)
        # Time slots (duration/delay) are ALWAYS stored in ns: binding rewrites the field to
        # its "str (ns)" (ns) display, so a slot left in us/ms would scan in that unit while
        # the card shows "str (ns)" -- a silent 1000x mismatch.  Convert the nominal from the
        # field's entry unit to ns and pin the slot unit to ns so the Scan tab, the period/
        # delay card, and the compiled program all agree.  (DAC slots keep their raw "value"
        # unit -- a DAC code is not a time.)
        if kind in ("duration", "delay") and unit not in ("ns", "value"):
            nominal = float(nominal) * UNITS_TO_NS.get(unit, 1.0)
            unit = "ns"
        index = len(self.scan_slots)
        slot = ScanSlot(kind=kind, target=str(target), label=label, unit=unit, nominal=float(nominal))
        self.scan_slots.append(slot)
        self._apply_slot_binding(index, slot)
        for row in self.scan_table:
            row.append(float(nominal))
        self.validate()
        return index

    def unbind_slot(self, index: int, *, restore: float | str | None = None) -> "PulseTableState":
        """Remove scan slot ``index``; later slots shift down (s2 -> s1, ...)."""

        index = int(index)
        if index < 0 or index >= len(self.scan_slots):
            raise ValueError("scan slot index out of range.")
        self._clear_slot_binding(index, restore)
        del self.scan_slots[index]
        for row in self.scan_table:
            if index < len(row):
                del row[index]
        # Renumber: re-apply each remaining slot's variable to its field.
        self._renumber_slot_bindings()
        self.validate()
        return self

    def slot_index_for(self, kind: str, target: str) -> int | None:
        for index, slot in enumerate(self.scan_slots):
            if slot.kind == kind and slot.target == str(target):
                return index
        return None

    def _apply_slot_binding(self, index: int, slot: ScanSlot) -> None:
        var = slot_var(index)
        if slot.kind == "duration":
            period_index = int(slot.target)
            period = self.periods[period_index]
            self.periods[period_index] = PulsePeriod(var, period.states, unit="str (ns)", name=period.name)
        elif slot.kind == "delay":
            self.delays[slot.target] = var
            self.delay_units[slot.target] = "str (ns)"
        elif slot.kind == "dac":
            bus, period_index = slot.dac_bus, slot.dac_period
            plan = self.analog_bus_plan(bus)
            plan[period_index] = {"mode": "edge", "value": var}
            self.analog_bus_modes[bus] = plan

    def _clear_slot_binding(self, index: int, restore: float | str | None) -> None:
        slot = self.scan_slots[index]
        var = slot_var(index)
        if slot.kind == "duration":
            period_index = int(slot.target)
            period = self.periods[period_index]
            if str(period.duration) == var:
                value = 1_000 if restore is None else restore
                self.periods[period_index] = PulsePeriod(value, period.states, unit="ns", name=period.name)
        elif slot.kind == "delay":
            if str(self.delays.get(slot.target)) == var:
                self.delays[slot.target] = 0 if restore is None else restore
                self.delay_units[slot.target] = "ns"
        elif slot.kind == "dac":
            bus, period_index = slot.dac_bus, slot.dac_period
            plan = self.analog_bus_plan(bus)
            if period_index < len(plan) and str(plan[period_index].get("value")) == var:
                plan[period_index] = {"mode": "hold", "value": None}
                self.analog_bus_modes[bus] = plan

    def _renumber_slot_bindings(self) -> None:
        # Map any field referencing an old s{k} to its new index after a removal.
        for new_index, slot in enumerate(self.scan_slots):
            self._apply_slot_binding(new_index, slot)

    def set_scan_table(self, rows: Sequence[Sequence[float]]) -> "PulseTableState":
        self.scan_table = _normalize_scan_table(rows, n_slots=len(self.scan_slots))
        self.validate()
        return self

    def load_scan_table(self, path: str | Path) -> "PulseTableState":
        return self.set_scan_table(load_scan_table(path))

    def with_slots_resolved(self, slots: Mapping[str, float]) -> "PulseTableState":
        """Return a non-scan copy with each slot replaced by a constant value.

        Time slots take ns values; ``dac`` slots take integer DAC codes.  Used
        for terse single-point notebook scans where one value is set per shot.
        """

        new = PulseTableState.from_dict(self.to_dict())
        for index, slot in enumerate(new.scan_slots):
            value = float(slots.get(slot_var(index), 0.0))
            if slot.kind == "duration":
                period_index = int(slot.target)
                period = new.periods[period_index]
                new.periods[period_index] = PulsePeriod(value, period.states, unit="ns", name=period.name)
            elif slot.kind == "delay":
                new.delays[slot.target] = value
                new.delay_units[slot.target] = "ns"
            elif slot.kind == "dac":
                plan = new.analog_bus_plan(slot.dac_bus)
                plan[slot.dac_period] = {"mode": "edge", "value": int(round(value))}
                new.analog_bus_modes[slot.dac_bus] = plan
        new.scan_slots = []
        new.scan_table = []
        new.apply_analog_bus_modes_to_period_states()
        new.validate()
        return new

    def primary_time_slot(self) -> str | None:
        """Return the variable name of the first duration/delay scan slot."""

        for index, slot in enumerate(self.scan_slots):
            if slot.is_time:
                return slot_var(index)
        return None

    def validate(self, *, slots: Mapping[str, float] | None = None, time_step_ns: float | None = None,
                 validate_scan_slots: bool = True) -> "PulseTableState":
        # ``validate_scan_slots`` checks the slot bindings + the FULL N-row scan table; it is
        # SLOT-INDEPENDENT, so a per-scan-point validate (compile_scan) sets it False after
        # one full check -- otherwise validating N points each rescans the whole table, an
        # O(N^2) blow-up that dominated compile at thousands of points.
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        slots = self.reference_slots() if slots is None else dict(slots)
        if not self.channels:
            raise ValueError("pulse table must have at least one channel.")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("pulse table channels must be unique.")
        if len(set(self.visible_channels)) != len(self.visible_channels):
            raise ValueError("visible channels must be unique.")
        if not self.visible_channels:
            raise ValueError("pulse table must show at least one channel.")
        unknown_visible = [channel for channel in self.visible_channels if channel not in self.channels]
        if unknown_visible:
            raise ValueError(f"visible channels are not in hardware channels: {unknown_visible}.")
        unknown_labels = [channel for channel in self.channel_labels if channel not in self.channels]
        if unknown_labels:
            raise ValueError(f"channel label keys are not in hardware channels: {unknown_labels}.")
        bus_members: list[str] = []
        for name, members in self.analog_buses.items():
            if not members:
                raise ValueError(f"analog bus {name!r} must contain at least one channel.")
            unknown_members = [channel for channel in members if channel not in self.channels]
            if unknown_members:
                raise ValueError(f"analog bus {name!r} contains channels not in hardware channels: {unknown_members}.")
            bus_members.extend(members)
        duplicated_bus_members = sorted({channel for channel in bus_members if bus_members.count(channel) > 1})
        if duplicated_bus_members:
            raise ValueError(f"analog bus channels must not overlap: {duplicated_bus_members}.")
        known_buses = self.bus_channels(min_width=1)
        unknown_bus_modes = [name for name in self.analog_bus_modes if name not in known_buses]
        if unknown_bus_modes:
            raise ValueError(f"analog bus modes reference unknown buses: {unknown_bus_modes}.")
        width = len(self.channels)
        if not self.periods:
            raise ValueError("pulse table must have at least one period.")
        for index, period in enumerate(self.periods):
            if len(period.states) != width:
                raise ValueError(f"period {index} has {len(period.states)} states but {width} channels.")
            for value in period.states:
                if int(value) not in (0, 1):
                    raise ValueError("period states must be 0 or 1.")
            period.duration_steps(slots=slots, time_step_ns=step_ns)
        for bus_name, entries in self.analog_bus_modes.items():
            members = known_buses[bus_name]
            if len(entries) != len(self.periods):
                raise ValueError(f"analog bus {bus_name!r} has {len(entries)} mode entries but {len(self.periods)} periods.")
            max_value = (1 << len(members)) - 1
            for index, entry in enumerate(entries):
                mode = str(entry.get("mode", "hold")).lower()
                if mode not in ANALOG_BUS_MODES:
                    raise ValueError(f"analog bus {bus_name!r} period {index} has unsupported mode {mode!r}.")
                value = entry.get("value")
                if mode == "hold":
                    if value is not None:
                        raise ValueError(f"analog bus {bus_name!r} period {index} hold mode must not have a value.")
                    continue
                if value is None:
                    raise ValueError(f"analog bus {bus_name!r} period {index} {mode} mode requires a value.")
                if _is_slot_ref(value):
                    continue
                value_int = int(value)
                if value_int < 0 or value_int > max_value:
                    raise ValueError(f"analog bus {bus_name!r} period {index} value must be between 0 and {max_value}.")
        if self.repeat_count < 1:
            raise ValueError("repeat_count must be >= 1.")
        if (self.repeat_start is None) != (self.repeat_end is None):
            raise ValueError("repeat_start and repeat_end must be set together.")
        if self.repeat_start is not None and self.repeat_end is not None:
            if self.repeat_start < 0 or self.repeat_end < self.repeat_start or self.repeat_end >= len(self.periods):
                raise ValueError("repeat bracket must select an existing period range.")
        for channel, delay in self.delays.items():
            if channel not in self.channels:
                raise ValueError(f"delay channel {channel!r} is not in channels.")
            self.delay_steps(channel, slots=slots, time_step_ns=step_ns)
        if validate_scan_slots:
            self._validate_scan_slots()
        return self

    def _validate_scan_slots(self) -> None:
        for index, slot in enumerate(self.scan_slots):
            if slot.kind == "duration":
                period_index = int(slot.target) if slot.target.lstrip("-").isdigit() else -1
                if period_index < 0 or period_index >= len(self.periods):
                    raise ValueError(f"scan slot {index} binds duration of missing period {slot.target!r}.")
            elif slot.kind == "delay":
                if slot.target not in self.channels:
                    raise ValueError(f"scan slot {index} binds delay of unknown channel {slot.target!r}.")
            elif slot.kind == "dac":
                if slot.dac_bus not in self.bus_channels(min_width=1):
                    raise ValueError(f"scan slot {index} binds unknown analog bus {slot.dac_bus!r}.")
                if slot.dac_period < 0 or slot.dac_period >= len(self.periods):
                    raise ValueError(f"scan slot {index} binds dac of missing period {slot.dac_period}.")
        for row_index, row in enumerate(self.scan_table):
            if len(row) != len(self.scan_slots):
                raise ValueError(
                    f"scan table row {row_index} has {len(row)} values but {len(self.scan_slots)} slots."
                )

    def _normalize_analog_bus_modes(
        self,
        payload: Mapping[str, Sequence[Mapping[str, object] | str | None]] | None,
    ) -> dict[str, list[dict[str, object]]]:
        out: dict[str, list[dict[str, object]]] = {}
        for bus_name, entries in dict(payload or {}).items():
            normalized: list[dict[str, object]] = []
            for item in list(entries):
                if item is None:
                    normalized.append({"mode": "hold", "value": None})
                elif isinstance(item, str):
                    mode = item.strip().lower()
                    normalized.append({"mode": mode, "value": None})
                else:
                    entry = dict(item)
                    mode = str(entry.get("mode", "hold")).strip().lower()
                    value = entry.get("value")
                    normalized.append({"mode": mode, "value": None if mode == "hold" else _coerce_bus_value(value)})
            while len(normalized) < len(self.periods):
                normalized.append({"mode": "hold", "value": None})
            out[str(bus_name)] = normalized[: len(self.periods)]
        return out

    def aligned_to_channels(self, channels: Sequence[str]) -> "PulseTableState":
        """Return a copy whose channel list matches hardware, filling missing channels off."""

        channels = list(channel_names(channels, "channels"))
        source_index = {channel: index for index, channel in enumerate(self.channels)}
        unknown = [channel for channel in self.channels if channel not in source_index or channel not in channels]
        if unknown:
            raise ValueError(f"pulse state channels are not in hardware channels: {unknown}.")
        periods = []
        for period in self.periods:
            states = tuple(int(period.states[source_index[channel]]) if channel in source_index else 0 for channel in channels)
            periods.append(PulsePeriod(period.duration, states, unit=period.unit, name=period.name))
        visible = [channel for channel in self.visible_channels if channel in channels]
        if not visible:
            visible = default_visible_channels(channels)
        return type(self)(
            channels=channels,
            periods=periods,
            delays={channel: value for channel, value in self.delays.items() if channel in channels},
            delay_units={channel: value for channel, value in self.delay_units.items() if channel in channels},
            name=self.name,
            scan_slots=[slot.to_dict() for slot in self.scan_slots],
            scan_table=[list(row) for row in self.scan_table],
            time_step_ns=self.time_step_ns,
            repeat_start=self.repeat_start,
            repeat_end=self.repeat_end,
            repeat_count=self.repeat_count,
            repeat_forever=self.repeat_forever,
            visible_channels=visible,
            channel_labels={channel: value for channel, value in self.channel_labels.items() if channel in channels},
            analog_buses={
                name: filtered
                for name, members in self.analog_buses.items()
                for filtered in ([channel for channel in members if channel in channels],)
                if filtered
            },
            analog_bus_modes={
                name: list(entries)
                for name, entries in self.analog_bus_modes.items()
                if name in self.bus_channels(min_width=1)
            },
        )

    def label_for(self, channel: str) -> str:
        channel = self.channels[self.channel_index(channel)]
        return self.channel_labels.get(channel) or channel

    def channel_index(self, channel: str) -> int:
        try:
            return self.channels.index(str(channel))
        except ValueError as exc:
            raise ValueError(f"unknown channel {channel!r}.") from exc

    def active_channels(self) -> list[str]:
        active: list[str] = []
        for channel in self.channels:
            if channel in self.period_active_channels() or self.delay_steps(channel) != 0:
                active.append(channel)
        return active

    def period_active_channels(self) -> list[str]:
        active: list[str] = []
        for channel in self.channels:
            index = self.channel_index(channel)
            if any(int(period.states[index]) for period in self.periods):
                active.append(channel)
        return active

    def hidden_active_channels(self) -> list[str]:
        visible = set(self.visible_channels)
        return [channel for channel in self.period_active_channels() if channel not in visible]

    def repeat_forever_boundary_active_channels(self) -> list[str]:
        """Channels that go high when an internal finite bracket restarts the table."""

        if not self.repeat_forever or self.repeat_start is None or self.repeat_end is None or self.repeat_count <= 1:
            return []
        if int(self.repeat_start) == 0 and int(self.repeat_end) == len(self.periods) - 1:
            return []
        first_states = self.periods[0].states
        return [channel for channel, state in zip(self.channels, first_states) if int(state)]

    def bus_channels(self, *, min_width: int = 2) -> dict[str, list[str]]:
        """Return logical bus channels inferred from labels like ``da[0]``."""

        explicit = {
            str(name): [channel for channel in members if channel in self.channels]
            for name, members in self.analog_buses.items()
            if len([channel for channel in members if channel in self.channels]) >= int(min_width)
        }
        inferred = infer_bus_channels(self.channels, self.channel_labels, min_width=min_width)
        inferred.update(explicit)
        return inferred

    def bus_value(self, period_index: int, bus_name: str) -> int:
        """Return the integer value encoded by a bus in one period."""

        period = self.periods[int(period_index)]
        groups = self.bus_channels()
        if bus_name not in groups:
            raise ValueError(f"unknown bus channel {bus_name!r}.")
        value = 0
        for bit, channel in enumerate(groups[bus_name]):
            if int(period.states[self.channel_index(channel)]):
                value |= 1 << bit
        return value

    def set_bus_value(self, period_index: int, bus_name: str, value: int) -> "PulseTableState":
        """Set one logical bus value, updating its underlying TTL bit channels."""

        period_index = int(period_index)
        groups = self.bus_channels()
        if bus_name not in groups:
            raise ValueError(f"unknown bus channel {bus_name!r}.")
        members = groups[bus_name]
        max_value = (1 << len(members)) - 1
        value = int(value)
        if value < 0 or value > max_value:
            raise ValueError(f"{bus_name} must be between 0 and {max_value}.")
        period = self.periods[period_index]
        states = list(period.states)
        for bit, channel in enumerate(members):
            states[self.channel_index(channel)] = 1 if (value >> bit) & 1 else 0
        self.periods[period_index] = PulsePeriod(period.duration, tuple(states), unit=period.unit, name=period.name)
        self.set_analog_bus_mode(period_index, bus_name, "edge", value=value, validate=False)
        self.validate()
        return self

    def analog_bus_plan(self, bus_name: str) -> list[dict[str, object]]:
        """Return one normalized ``hold/edge/ramp`` entry per period."""

        bus_name = str(bus_name)
        groups = self.bus_channels(min_width=1)
        if bus_name not in groups:
            raise ValueError(f"unknown bus channel {bus_name!r}.")
        if bus_name in self.analog_bus_modes:
            return [dict(item) for item in self.analog_bus_modes[bus_name]]
        out: list[dict[str, object]] = []
        previous: int | None = None
        for index in range(len(self.periods)):
            value = self.bus_value(index, bus_name)
            if index == 0 or previous is None or value != previous:
                out.append({"mode": "edge", "value": int(value)})
            else:
                out.append({"mode": "hold", "value": None})
            previous = value
        return out

    def set_analog_bus_mode(
        self,
        period_index: int,
        bus_name: str,
        mode: str,
        *,
        value: int | None = None,
        validate: bool = True,
    ) -> "PulseTableState":
        """Set one bus period mode and optional value."""

        period_index = int(period_index)
        bus_name = str(bus_name)
        mode = str(mode).strip().lower()
        if mode not in ANALOG_BUS_MODES:
            raise ValueError(f"analog bus mode must be one of {ANALOG_BUS_MODES}.")
        plan = self.analog_bus_plan(bus_name)
        if period_index < 0 or period_index >= len(plan):
            raise ValueError("period_index is out of range.")
        if mode == "hold":
            plan[period_index] = {"mode": "hold", "value": None}
        else:
            if value is None:
                value = self.bus_value(period_index, bus_name)
            plan[period_index] = {"mode": mode, "value": _coerce_bus_value(value)}
        self.analog_bus_modes[bus_name] = plan
        if validate:
            self.apply_analog_bus_modes_to_period_states()
            self.validate()
        return self

    def apply_analog_bus_modes_to_period_states(self) -> "PulseTableState":
        """Project logical bus mode/value rows back to underlying TTL bits.

        Slot-referenced (scanned) bus values use their reference scan point so
        the underlying TTL bits keep a sensible preview value.
        """

        slots = self.reference_slots()
        starts = self._period_start_steps(slots=slots, time_step_ns=self.time_step_ns)
        groups = self.bus_channels(min_width=1)
        for bus_name, members in groups.items():
            plan = self._resolved_bus_plan(bus_name, slots)
            for period_index, period in enumerate(self.periods):
                value = _analog_bus_value_at_tick(plan, starts, starts[period_index])
                states = list(period.states)
                for bit, channel in enumerate(members):
                    states[self.channel_index(channel)] = 1 if (int(value) >> bit) & 1 else 0
                self.periods[period_index] = PulsePeriod(period.duration, tuple(states), unit=period.unit, name=period.name)
        return self

    def _resolved_bus_plan(self, bus_name: str, slots: Mapping[str, float]) -> list[dict[str, object]]:
        """Bus plan with slot references resolved to integers for preview."""

        plan = self.analog_bus_plan(bus_name)
        resolved: list[dict[str, object]] = []
        for entry in plan:
            value = entry.get("value")
            if _is_slot_ref(value):
                value = int(round(float(slots.get(str(value), 0.0))))
            resolved.append({"mode": entry.get("mode", "hold"), "value": value})
        return resolved

    def analog_bus_value_at_period_start(self, period_index: int, bus_name: str) -> int:
        slots = self.reference_slots()
        starts = self._period_start_steps(slots=slots, time_step_ns=self.time_step_ns)
        return _analog_bus_value_at_tick(self._resolved_bus_plan(bus_name, slots), starts, starts[int(period_index)])

    def show_channel(self, channel: str, *, index: int | None = None) -> "PulseTableState":
        channel = self.channels[self.channel_index(channel)]
        if channel in self.visible_channels:
            return self
        if index is None:
            target = self.channel_index(channel)
            index = sum(1 for item in self.visible_channels if self.channel_index(item) < target)
            self.visible_channels.insert(index, channel)
        else:
            self.visible_channels.insert(max(0, min(int(index), len(self.visible_channels))), channel)
        self.validate()
        return self

    def hide_channel(self, channel: str, *, clear: bool = False) -> "PulseTableState":
        channel = self.channels[self.channel_index(channel)]
        if channel not in self.visible_channels:
            return self
        if channel in self.period_active_channels():
            if not clear:
                raise ValueError(f"channel {channel!r} is active; pass clear=True before hiding it.")
            self.clear_channel(channel)
        self.visible_channels = [item for item in self.visible_channels if item != channel]
        self.validate()
        return self

    def clear_channel(self, channel: str, *, clear_delay: bool = False) -> "PulseTableState":
        index = self.channel_index(channel)
        channel = self.channels[index]
        self.periods = [
            PulsePeriod(period.duration, tuple(0 if i == index else value for i, value in enumerate(period.states)), unit=period.unit, name=period.name)
            for period in self.periods
        ]
        if clear_delay:
            self.delays.pop(channel, None)
            self.delay_units.pop(channel, None)
        self.validate()
        return self

    def set_period_state(self, period_index: int, channel: str, value: int) -> "PulseTableState":
        period_index = int(period_index)
        if period_index < 0 or period_index >= len(self.periods):
            raise ValueError("period_index is out of range.")
        channel_index = self.channel_index(channel)
        period = self.periods[period_index]
        states = list(period.states)
        states[channel_index] = 1 if int(value) else 0
        self.periods[period_index] = PulsePeriod(period.duration, tuple(states), unit=period.unit, name=period.name)
        self.validate()
        return self

    def expanded_periods(self) -> list[PulsePeriod]:
        if self.repeat_start is None or self.repeat_end is None or self.repeat_count == 1:
            return list(self.periods)
        return (
            list(self.periods[: self.repeat_start])
            + list(self.periods[self.repeat_start : self.repeat_end + 1]) * self.repeat_count
            + list(self.periods[self.repeat_end + 1 :])
        )

    def _expand_bracket_index(self, period_index: int) -> int:
        """Map an ORIGINAL period index to its FIRST index after the bracket is
        unrolled flat (see :meth:`unrolled_bracket`).

        Periods before the bracket keep their index; a bracketed period maps to its
        first unrolled copy (so a scanned-duration/scanned-DAC slot whose ``target``
        names it still points at a period that carries the ``sN`` expression -- and
        because every copy carries that same expression, every copy scans together);
        a period after the bracket shifts by ``(repeat_count-1) * bracket_length``.
        """

        rs, re, rc = int(self.repeat_start), int(self.repeat_end), int(self.repeat_count)
        if period_index <= re:
            return period_index                      # before or first copy of the bracket
        loop_len = re - rs + 1
        return period_index + (rc - 1) * loop_len     # after the bracket

    def unrolled_bracket(self) -> "PulseTableState":
        """Return a NEW state with the inner finite repeat bracket fully UNROLLED into a
        flat period list (``repeat_count`` becomes 1, the bracket is cleared).

        This is the unifying trick that makes a (constant OR scanned) channel delay work
        with an inner bracket: once the bracket is flat there is no inner-loop boundary
        for an additively-shifted edge to cross, so the existing flat machinery (additive
        delay + reordering delay lanes + affine scan + repeat_forever) handles delays in
        ANY form.  The whole flat frame can still repeat via ``repeat_forever``.

        Each bracketed ``PulsePeriod`` is duplicated to its new indices -- carrying its
        duration expression (incl. ``sN``), states, unit and name automatically -- and so
        is each analog-bus plan entry (mode + value, incl. a scanned ``sN`` DAC level), so
        a scanned duration/DAC of a bracketed period scans every copy in lockstep.
        Per-channel delays, scan slots and the scan table copy unchanged; only the slot
        ``target`` period indices are remapped to stay valid.  No bracket -> ``self``.
        """

        if self.repeat_start is None or self.repeat_end is None or int(self.repeat_count) <= 1:
            return self
        rs, re = int(self.repeat_start), int(self.repeat_end)

        def expand(items: Sequence) -> list:
            return list(items[:rs]) + list(items[rs : re + 1]) * int(self.repeat_count) + list(items[re + 1 :])

        scan_slots: list[dict[str, object]] = []
        for slot in self.scan_slots:
            payload = slot.to_dict()
            if slot.kind == "duration" and str(slot.target).lstrip("-").isdigit():
                payload["target"] = str(self._expand_bracket_index(int(slot.target)))
            elif slot.kind == "dac":
                bus, period_index = slot.target.split("@", 1)
                payload["target"] = f"{bus}@{self._expand_bracket_index(int(period_index))}"
            scan_slots.append(payload)

        return type(self)(
            channels=list(self.channels),
            periods=[PulsePeriod(p.duration, p.states, unit=p.unit, name=p.name) for p in expand(self.periods)],
            delays=dict(self.delays),
            delay_units=dict(self.delay_units),
            name=self.name,
            scan_slots=scan_slots,
            scan_table=[list(row) for row in self.scan_table],
            time_step_ns=self.time_step_ns,
            repeat_start=None,
            repeat_end=None,
            repeat_count=1,
            repeat_forever=self.repeat_forever,
            visible_channels=list(self.visible_channels),
            channel_labels=dict(self.channel_labels),
            analog_buses={name: list(members) for name, members in self.analog_buses.items()},
            analog_bus_modes={name: [dict(entry) for entry in expand(entries)] for name, entries in self.analog_bus_modes.items()},
        )

    def delay_steps(self, channel: str, *, slots: Mapping[str, float] | None = None, time_step_ns: float | None = None) -> int:
        raw = self.delays.get(channel, 0.0)
        unit = self.delay_units.get(channel, "ns")
        if unit not in UNITS_TO_NS:
            raise ValueError(f"unsupported delay unit {unit!r}.")
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        return quantized_time_steps(
            eval_time_expr(raw, slots=slots) * UNITS_TO_NS[unit],
            time_step_ns=step_ns,
            name=f"delay for {channel!r}",
            allow_zero=True,
            allow_negative=True,
        )

    def delay_ns(self, channel: str, *, slots: Mapping[str, float] | None = None, time_step_ns: float | None = None) -> float:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        return self.delay_steps(channel, slots=slots, time_step_ns=step_ns) * step_ns

    def total_duration_steps(self, *, slots: Mapping[str, float] | None = None, time_step_ns: float | None = None) -> int:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        slots = self.reference_slots() if slots is None else dict(slots)
        return sum(period.duration_steps(slots=slots, time_step_ns=step_ns) for period in self.expanded_periods())

    def total_duration_ns(self, *, slots: Mapping[str, float] | None = None, time_step_ns: float | None = None) -> float:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        return self.total_duration_steps(slots=slots, time_step_ns=step_ns) * step_ns

    def _period_start_steps(self, *, slots: Mapping[str, float] | None = None, time_step_ns: float) -> list[int]:
        starts = [0]
        for period in self.periods:
            starts.append(starts[-1] + period.duration_steps(slots=slots, time_step_ns=time_step_ns))
        return starts

    def to_sequence(
        self,
        *,
        name: str | None = None,
        slots: Mapping[str, float] | None = None,
        time_step_ns: float | None = None,
        expand_repeat: bool = True,
    ) -> PulseSequence:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        slots = self.reference_slots() if slots is None else dict(slots)
        self.validate(slots=slots, time_step_ns=step_ns)
        sequence = PulseSequence(name=name or self.name)
        # First build each channel's UN-delayed ON intervals (in steps) over the frame.
        starts: dict[str, int | None] = {channel: None for channel in self.channels}
        intervals: dict[str, list[tuple[int, int]]] = {channel: [] for channel in self.channels}
        t_steps = 0
        periods = self.expanded_periods() if expand_repeat else list(self.periods)
        for period in periods:
            next_t_steps = t_steps + period.duration_steps(slots=slots, time_step_ns=step_ns)
            for channel, state in zip(self.channels, period.states):
                active_start = starts[channel]
                if state and active_start is None:
                    starts[channel] = t_steps
                elif not state and active_start is not None:
                    intervals[channel].append((active_start, t_steps))
                    starts[channel] = None
            t_steps = next_t_steps
        for channel, active_start in starts.items():
            if active_start is not None:
                intervals[channel].append((active_start, t_steps))
        # Apply each channel's delay as a CYCLIC rotation within the frame
        # (delay %% total_duration): a pulse pushed past the frame end wraps to the
        # start.  This is the periodic ("inf") view used FOR THE PREVIEW ONLY -- it is a
        # convenient steady-state picture (matches Confocal-GUIv2's delay()).  The
        # HARDWARE delay is ADDITIVE / period-preserving (zero output before fire, no
        # wrap-in tail) -- see _pulse_table_edge_table in sequencer.py -- so this cyclic
        # view is NOT what the streamer plays; do not use to_sequence as the hardware
        # truth.  See _cyclic_shift_interval.
        total_steps = t_steps
        for channel in self.channels:
            d_steps = self.delay_steps(channel, slots=slots, time_step_ns=step_ns)
            for start_steps, stop_steps in intervals[channel]:
                for a, b in _cyclic_shift_interval(start_steps, stop_steps, d_steps, total_steps):
                    if b > a:
                        sequence = sequence.pulse(channel, a * step_ns * 1e-9, (b - a) * step_ns * 1e-9)
        return sequence

    def compile(
        self,
        *,
        clock_hz: float,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        slots: Mapping[str, float] | None = None,
        repeat_forever: bool | None = None,
    ):
        from ..devices.sequencer import compile_pulse_table_runtime_program

        clock_hz = positive_float(clock_hz, "clock_hz")
        return compile_pulse_table_runtime_program(
            self,
            channels=self.channels,
            clock_hz=clock_hz,
            trigger_channels=trigger_channels,
            slots=slots,
            repeat_forever=self.repeat_forever if repeat_forever is None else bool(repeat_forever),
        )

    def compile_scan(
        self,
        *,
        clock_hz: float,
        scan_table: Sequence[Sequence[float]] | None = None,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        repeat_forever: bool | None = None,
    ):
        from ..devices.sequencer import compile_pulse_table_scan_runtime_program

        clock_hz = positive_float(clock_hz, "clock_hz")
        return compile_pulse_table_scan_runtime_program(
            self,
            channels=self.channels,
            clock_hz=clock_hz,
            trigger_channels=trigger_channels,
            scan_table=self.scan_table if scan_table is None else scan_table,
            repeat_forever=self.repeat_forever if repeat_forever is None else bool(repeat_forever),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "name": self.name,
            "channels": list(self.channels),
            "scan_slots": [slot.to_dict() for slot in self.scan_slots],
            "scan_table": [list(row) for row in self.scan_table],
            "time_step_ns": self.time_step_ns,
            "periods": [period.to_dict() for period in self.periods],
            "visible_channels": list(self.visible_channels),
            "channel_labels": dict(self.channel_labels),
            "analog_buses": {name: list(members) for name, members in self.analog_buses.items()},
            "analog_bus_modes": {
                name: [dict(entry) for entry in entries]
                for name, entries in self.analog_bus_modes.items()
            },
            "delays": dict(self.delays),
            "delay_units": dict(self.delay_units),
            "repeat_start": self.repeat_start,
            "repeat_end": self.repeat_end,
            "repeat_count": self.repeat_count,
            "repeat_forever": self.repeat_forever,
        }

    def snapped(self, *, time_step_ns: float | None = None) -> "PulseTableState":
        """Return a copy with every LITERAL time value snapped to the clock-tick grid:
        period durations up to ``>= 1`` tick, channel delays to the nearest tick (sign
        preserved), and scan-table points to the nearest tick (DAC points to the nearest
        integer code).  Slot EXPRESSIONS (``s0`` ...) are kept verbatim; the compiler
        snaps their affine base.  This is the single snap source shared by the GUI
        display and the server / pulse-transfer API, so what the user sees and what the
        hardware runs always agree (the hardware can only land on whole ticks)."""

        step = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        copy = PulseTableState.from_dict(self.to_dict())
        copy.periods = [
            PulsePeriod(
                _snap_literal_time_value(period.duration, period.unit, step, allow_zero=False),
                period.states,
                unit=period.unit,
                name=period.name,
            )
            for period in copy.periods
        ]
        copy.delays = {
            channel: _snap_literal_time_value(
                value, copy.delay_units.get(channel, "ns"), step, allow_zero=True, allow_negative=True
            )
            for channel, value in copy.delays.items()
        }
        copy.scan_table = snap_scan_table(copy.scan_table, copy.scan_slots, time_step_ns=step)
        copy.validate()
        return copy

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PulseTableState":
        if payload.get("schema", cls.schema) != cls.schema:
            raise ValueError("unsupported pulse table schema.")
        return cls(
            name=str(payload["name"]) if "name" in payload else None,
            channels=payload["channels"],
            scan_slots=payload.get("scan_slots", ()),
            scan_table=payload.get("scan_table", ()),
            time_step_ns=float(payload.get("time_step_ns", 1.0)),
            periods=[PulsePeriod.from_dict(item) for item in payload.get("periods", [])],
            visible_channels=payload.get("visible_channels"),
            channel_labels=dict(payload.get("channel_labels", {})),
            analog_buses=dict(payload.get("analog_buses", {})),
            analog_bus_modes=dict(payload.get("analog_bus_modes", {})),
            delays=dict(payload.get("delays", {})),
            delay_units=dict(payload.get("delay_units", {})),
            repeat_start=payload.get("repeat_start"),
            repeat_end=payload.get("repeat_end"),
            repeat_count=int(payload.get("repeat_count", 1)),
            repeat_forever=bool(payload.get("repeat_forever", True)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PulseTableState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_sequence(cls, sequence: PulseSequence, *, channels: Sequence[str], clock_hz: float = 50_000_000.0) -> "PulseTableState":
        ticks, masks, channels = sequence.edges(clock_hz=clock_hz, channels=channels)
        periods: list[PulsePeriod] = []
        if not ticks:
            return cls(channels=channels, name=sequence.name)
        if ticks[0] > 0:
            periods.append(PulsePeriod(duration=int(ticks[0]), unit="str (ns)", states=tuple(0 for _ in channels), name="idle"))
        for index, tick in enumerate(ticks):
            next_tick = ticks[index + 1] if index + 1 < len(ticks) else int(round(sequence.duration * clock_hz))
            duration_ticks = next_tick - tick
            if duration_ticks <= 0:
                continue
            states = tuple((masks[index] >> bit) & 1 for bit in range(len(channels)))
            periods.append(PulsePeriod(duration=int(duration_ticks), unit="str (ns)", states=states))
        visible = []
        for index, channel in enumerate(channels):
            if any((mask >> index) & 1 for mask in masks) or channel in sequence.delays:
                visible.append(channel)
        step_ns = 1e9 / positive_float(clock_hz, "clock_hz")
        scaled_periods = [
            PulsePeriod(duration=int(period.duration) * step_ns, unit="ns", states=period.states, name=period.name)
            for period in periods
        ]
        return cls(channels=channels, periods=scaled_periods, name=sequence.name, time_step_ns=step_ns, visible_channels=visible or default_visible_channels(channels))


def default_periods(channels: Sequence[str]) -> list[PulsePeriod]:
    width = len(channel_names(channels, "channels"))
    return [
        PulsePeriod(1_000, tuple(1 if index == 0 else 0 for index in range(width)), name=""),
        PulsePeriod(1_000, tuple(0 for _ in range(width)), name=""),
    ]


def default_visible_channels(channels: Sequence[str]) -> list[str]:
    channels = list(channel_names(channels, "channels"))
    preferred = [channel for channel in ("trap", "cooling", "probe", "emCCD") if channel in channels]
    if preferred:
        return preferred
    return channels[: min(8, len(channels))]


def infer_bus_channels(
    channels: Sequence[str],
    channel_labels: Mapping[str, str] | None = None,
    *,
    min_width: int = 2,
) -> dict[str, list[str]]:
    """Infer logical buses from labels such as ``da_dipole[0]`` ... ``[9]``."""

    labels = {str(k): str(v) for k, v in dict(channel_labels or {}).items()}
    by_base: dict[str, dict[int, str]] = {}
    for channel in channel_names(channels, "channels"):
        label = labels.get(channel) or channel
        match = BUS_LABEL_RE.fullmatch(str(label).strip())
        if not match:
            continue
        base = match.group("base").strip()
        bit = int(match.group("bit"))
        if not base:
            continue
        by_base.setdefault(base, {})[bit] = channel
    out: dict[str, list[str]] = {}
    for base, members in by_base.items():
        if len(members) < int(min_width):
            continue
        bits = sorted(members)
        if bits != list(range(bits[0], bits[-1] + 1)):
            continue
        out[base] = [members[bit] for bit in bits]
    return out


def _analog_bus_value_at_tick(plan: Sequence[Mapping[str, object]], starts: Sequence[int], tick: int) -> int:
    tick = int(tick)
    anchors: list[tuple[int, int, str, int]] = []
    for index, entry in enumerate(plan):
        mode = str(entry.get("mode", "hold")).lower()
        if mode not in {"edge", "ramp"}:
            continue
        value = entry.get("value")
        if value is None:
            continue
        anchors.append((index, int(starts[index]), mode, int(value)))
    if not anchors:
        return 0
    first = anchors[0]
    if tick < first[1]:
        return 0
    previous = first
    for anchor in anchors[1:]:
        _index, anchor_tick, mode, value = anchor
        if tick < anchor_tick:
            if mode == "ramp" and anchor_tick > previous[1]:
                fraction = (tick - previous[1]) / (anchor_tick - previous[1])
                return int(round(previous[3] + (value - previous[3]) * fraction))
            return int(previous[3])
        previous = anchor
    return int(previous[3])


def _is_slot_ref(value: object) -> bool:
    return isinstance(value, str) and bool(SLOT_VAR_RE.fullmatch(value.strip()))


def _coerce_bus_value(value: object) -> object:
    if value is None:
        return None
    if _is_slot_ref(value):
        return value.strip()
    return int(value)


def _normalize_scan_table(rows: Sequence[Sequence[float]] | None, *, n_slots: int) -> list[list[float]]:
    if rows is None:
        return []
    out: list[list[float]] = []
    for index, row in enumerate(rows):
        if isinstance(row, Number):
            values = [float(row)]
        else:
            values = [float(value) for value in row]
        if n_slots and len(values) != n_slots:
            if len(values) < n_slots:
                values = values + [0.0] * (n_slots - len(values))
            else:
                raise ValueError(f"scan table row {index} has {len(values)} values but {n_slots} slots.")
        out.append(values)
    return out


def load_scan_table(path: str | Path) -> list[list[float]]:
    """Load a scan table (``N_points x N_slots``) from ``.npy``/``.csv``/``.txt``.

    ``.npy`` is read with NumPy.  Text files accept comma or whitespace
    separators and ignore ``#`` comment lines and a single header line of names.
    """

    import numpy as np

    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    elif path.suffix.lower() == ".json":
        array = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
    else:
        text = path.read_text(encoding="utf-8")
        delimiter = "," if "," in text else None
        rows = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(delimiter) if delimiter else stripped.split()
            try:
                rows.append([float(part) for part in parts])
            except ValueError:
                continue  # header / names line
        array = np.asarray(rows, dtype=float) if rows else np.zeros((0, 0))
    array = np.atleast_2d(np.asarray(array, dtype=float))
    return [[float(value) for value in row] for row in array]


def eval_time_expr(value: float | str, *, slots: Mapping[str, float] | None = None) -> float:
    """Evaluate a numeric expression with scan-slot variables ``s0, s1, ...`` (ns)."""

    if isinstance(value, Number):
        out = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("time expression must not be empty.")
        out = _SafeEval(slots).eval(text)
    if not math.isfinite(out):
        raise ValueError("time expression must be finite.")
    return out


def positive_time_step_ns(value: float | str) -> float:
    out = eval_time_expr(value, slots=None)
    if out <= 0:
        raise ValueError("time_step_ns must be > 0.")
    return out


def quantized_time_steps(
    value_ns: float | str,
    *,
    time_step_ns: float,
    name: str,
    allow_zero: bool,
    allow_negative: bool = False,
) -> int:
    value = eval_time_expr(value_ns, slots=None)
    step = positive_time_step_ns(time_step_ns)
    raw_steps = value / step
    # Snap to the nearest tick (ties away from zero), mirroring the confocal
    # align_to_resolution semantics.  An off-grid value is NEVER rejected -- the
    # hardware clock can only land on whole ticks, so we quietly round.
    steps = int(math.floor(raw_steps + 0.5)) if raw_steps >= 0 else int(math.ceil(raw_steps - 0.5))
    if steps < 0 and not allow_negative:
        steps = 0
    if steps == 0 and not allow_zero:
        # A period duration must occupy at least one tick (>= time_step_ns).
        # Snap *up* to one tick instead of rejecting, so e.g. 5 ns -> 20 ns.
        steps = 1
    return steps


def _snap_literal_time_value(
    value: float | str,
    unit: str,
    time_step_ns: float,
    *,
    allow_zero: bool,
    allow_negative: bool = False,
) -> float | str:
    """Snap one literal time value (in ``unit``) to the clock-tick grid, returned in
    the SAME unit.  A scan-slot EXPRESSION (anything that is not a plain number, e.g.
    ``"s0"`` or ``"20+s0"``) is returned unchanged -- the compiler snaps its affine
    base instead, so binding/expressions are never corrupted."""

    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
    else:
        number = float(value)
    factor = UNITS_TO_NS.get(str(unit), 1.0)
    snapped_ns = quantized_time_ns(
        number * factor, time_step_ns=time_step_ns, name="time value",
        allow_zero=allow_zero, allow_negative=allow_negative,
    )
    out = snapped_ns / factor
    return int(out) if float(out).is_integer() else out


def snap_scan_table(
    scan_table: Sequence[Sequence[float]],
    scan_slots: Sequence["ScanSlot"],
    *,
    time_step_ns: float,
) -> list[list[float]]:
    """Snap every scan-table point to the clock grid: time slots to the nearest tick
    (in the slot's display unit, sign preserved), DAC slots to the nearest integer
    code.  One shared snap source for the GUI (so the displayed/saved table matches)
    and the server/pulse API (so the transferred pulse matches the hardware)."""

    step = positive_time_step_ns(time_step_ns)
    out: list[list[float]] = []
    for row in scan_table:
        new_row: list[float] = []
        for value, slot in zip(row, scan_slots):
            if getattr(slot, "kind", "") == "dac":
                new_row.append(float(int(round(float(value)))))
            else:
                snapped = _snap_literal_time_value(
                    float(value), slot.unit, step, allow_zero=True, allow_negative=True
                )
                new_row.append(float(snapped))
        out.append(new_row)
    return out


def quantized_time_ns(
    value_ns: float | str,
    *,
    time_step_ns: float,
    name: str,
    allow_zero: bool,
    allow_negative: bool = False,
) -> float:
    return quantized_time_steps(
        value_ns,
        time_step_ns=time_step_ns,
        name=name,
        allow_zero=allow_zero,
        allow_negative=allow_negative,
    ) * positive_time_step_ns(time_step_ns)


def affine_coeffs(
    value: float | str,
    *,
    slot_vars: Sequence[str],
    unit: str = "ns",
    time_step_ns: float = 1.0,
    coeff_frac_bits: int = 8,
) -> tuple[int, list[int]]:
    """Return ``(base_ticks, [coeff_fixed per slot var])`` for scan timing.

    The expression must be affine in the slot variables: ``c + sum(k_j * s_j)``.
    Coefficients are fixed-point with ``coeff_frac_bits`` fractional bits, scaled
    so the hardware tick is ``base + (sum(coeff_j * slot_tick_j) >> frac_bits)``.
    """

    if unit not in UNITS_TO_NS:
        raise ValueError(f"unsupported time unit {unit!r}.")
    base, coeff_map = _SafeEval(None).affine(value)
    unit_scale = UNITS_TO_NS[unit]
    step_ns = positive_time_step_ns(time_step_ns)
    base_ticks_raw = base * unit_scale / step_ns
    base_ticks = int(round(base_ticks_raw))
    if not math.isclose(base_ticks_raw, base_ticks, rel_tol=GRID_RTOL, abs_tol=GRID_ATOL_STEPS):
        raise ValueError(f"affine base {base * unit_scale:g} ns is not an integer multiple of time_step_ns={step_ns:g} ns.")
    unknown = [name for name in coeff_map if name not in slot_vars]
    if unknown:
        raise ValueError(f"expression references unbound scan variables {unknown}; bind them to slots first.")
    scale = 1 << int(coeff_frac_bits)
    coeffs: list[int] = []
    for name in slot_vars:
        coeff = coeff_map.get(name, 0.0)
        # slot ticks already carry the unit scale (values stored in ns -> ticks),
        # so coefficient is dimensionless * unit_scale to match base unit.
        fixed = int(round(coeff * unit_scale * scale))
        if not math.isclose(fixed / scale, coeff * unit_scale, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"coefficient {coeff:g} for {name} cannot be represented with {coeff_frac_bits} fractional bits.")
        coeffs.append(fixed)
    return base_ticks, coeffs


class _SafeEval:
    _binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }
    _unary = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}

    def __init__(self, slots: Mapping[str, float] | None = None):
        self.values = {str(k): float(v) for k, v in dict(slots or {}).items()}

    def eval(self, text: str) -> float:
        return float(self._visit(ast.parse(_insert_implicit_mul(text), mode="eval").body))

    def _visit(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.values:
                return self.values[node.id]
            if SLOT_VAR_RE.fullmatch(node.id):
                return self.values.get(node.id, 0.0)
        if isinstance(node, ast.BinOp) and type(node.op) in self._binops:
            return self._binops[type(node.op)](self._visit(node.left), self._visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary:
            return self._unary[type(node.op)](self._visit(node.operand))
        raise ValueError("time expression may only use numbers, scan slots s0.., +, -, *, /, **, and parentheses.")

    def affine(self, value: float | str) -> tuple[float, dict[str, float]]:
        if isinstance(value, Number):
            return float(value), {}
        text = str(value).strip()
        if not text:
            raise ValueError("time expression must not be empty.")
        return self._affine_visit(ast.parse(_insert_implicit_mul(text), mode="eval").body)

    def _affine_visit(self, node) -> tuple[float, dict[str, float]]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value), {}
        if isinstance(node, ast.Name) and SLOT_VAR_RE.fullmatch(node.id):
            return 0.0, {node.id: 1.0}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            base, coeffs = self._affine_visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -base, {name: -coeff for name, coeff in coeffs.items()}
            return base, coeffs
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left_base, left = self._affine_visit(node.left)
            right_base, right = self._affine_visit(node.right)
            sign = -1.0 if isinstance(node.op, ast.Sub) else 1.0
            merged = dict(left)
            for name, coeff in right.items():
                merged[name] = merged.get(name, 0.0) + sign * coeff
            return left_base + sign * right_base, merged
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left_base, left = self._affine_visit(node.left)
            right_base, right = self._affine_visit(node.right)
            if not left:
                return right_base * left_base, {name: coeff * left_base for name, coeff in right.items()}
            if not right:
                return left_base * right_base, {name: coeff * right_base for name, coeff in left.items()}
            raise ValueError("hardware scan timing only supports affine slot expressions; products of variables are not supported.")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left_base, left = self._affine_visit(node.left)
            right_base, right = self._affine_visit(node.right)
            if right or right_base == 0.0:
                raise ValueError("hardware scan timing only supports division by a nonzero constant.")
            return left_base / right_base, {name: coeff / right_base for name, coeff in left.items()}
        raise ValueError("hardware scan timing only supports affine expressions in scan slots s0...")


def _insert_implicit_mul(text: str) -> str:
    """Insert ``*`` for implicit multiplication before slot vars and parentheses."""

    var = r"(?:s\d+)"
    text = re.sub(r"(\d|\.|\))\s*(" + var + r")", r"\1*\2", text)
    text = re.sub(r"(" + var + r"|\)|\d|\.)\s*(\()", r"\1*\2", text)
    return text


__all__ = [
    "ANALOG_BUS_MODES",
    "SCAN_SLOT_KINDS",
    "PulsePeriod",
    "PulseTableState",
    "ScanSlot",
    "affine_coeffs",
    "default_pulse_name",
    "default_periods",
    "default_visible_channels",
    "eval_time_expr",
    "infer_bus_channels",
    "load_scan_table",
    "positive_time_step_ns",
    "quantized_time_ns",
    "quantized_time_steps",
    "slot_var",
    "snap_scan_table",
]
