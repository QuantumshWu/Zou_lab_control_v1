"""Confocal-style pulse-table editor model for ``PulseSequence``.

The model keeps the old GUI's user-facing idea: a horizontal list of periods,
where each period has one duration and a full digital state vector.  It does
not own hardware.  It only compiles to ``PulseSequence`` so notebooks, PyQt,
and remote FPGA sequencers share the same timing source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
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
SCAN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SCAN_HEADER_RE = re.compile(r"^\s*#\s*vars\s*:\s*(?P<vars>.+)$", re.IGNORECASE)
SCAN_TOKEN_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\((?P<unit>[^)]+)\)|\[(?P<bracket_unit>[^\]]+)\])?$")
ANALOG_BUS_MODES = ("hold", "edge", "ramp")
TIME_SCAN_UNITS = {
    "ns": 1.0,
    "us": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
}


@dataclass(frozen=True)
class ScanParameterTable:
    """One ordered scan table loaded from a text/JSON file."""

    names: tuple[str, ...]
    rows: tuple[dict[str, float], ...]
    units: dict[str, str]
    path: str = ""


def default_pulse_name() -> str:
    return "pulse_" + datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class PulsePeriod:
    """One period-card in the pulse GUI."""

    duration: float | str
    states: tuple[int, ...]
    unit: str = "ns"
    name: str = ""

    def duration_steps(
        self,
        *,
        x_ns: float = 0.0,
        y_ns: float = 0.0,
        time_step_ns: float = 1.0,
        variables: Mapping[str, float] | None = None,
    ) -> int:
        value = eval_time_expr(self.duration, x_ns=x_ns, y_ns=y_ns, variables=variables)
        unit = str(self.unit or "ns")
        if unit not in UNITS_TO_NS:
            raise ValueError(f"unsupported pulse duration unit {unit!r}.")
        return quantized_time_steps(value * UNITS_TO_NS[unit], time_step_ns=time_step_ns, name="period duration", allow_zero=False)

    def duration_ns(
        self,
        *,
        x_ns: float = 0.0,
        y_ns: float = 0.0,
        time_step_ns: float | None = None,
        variables: Mapping[str, float] | None = None,
    ) -> float:
        value = eval_time_expr(self.duration, x_ns=x_ns, y_ns=y_ns, variables=variables)
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

    def __init__(
        self,
        *,
        channels: Sequence[str],
        periods: Iterable[PulsePeriod] | None = None,
        delays: Mapping[str, float | str] | None = None,
        delay_units: Mapping[str, str] | None = None,
        name: str | None = None,
        x_ns: float = 0.0,
        y_ns: float = 0.0,
        scan_points: Sequence[Sequence[float] | float] | None = None,
        scan_variables: Mapping[str, float] | None = None,
        scan_bindings: Mapping[str, str] | None = None,
        scan_table_path: str | Path | None = None,
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
        self.x_ns = quantized_time_ns(x_ns, time_step_ns=self.time_step_ns, name="x_ns", allow_zero=True)
        self.y_ns = quantized_time_ns(y_ns, time_step_ns=self.time_step_ns, name="y_ns", allow_zero=True, allow_negative=True)
        if scan_points:
            raise ValueError("legacy x/y scan_points are no longer supported; link a named scan table file.")
        self.scan_points: list[tuple[float, float]] = []
        self.scan_variables = {
            normalize_scan_parameter_name(name): float(value)
            for name, value in dict(scan_variables or {}).items()
        }
        self.scan_bindings = {
            normalize_scan_binding_key(key): normalize_scan_parameter_name(value)
            for key, value in dict(scan_bindings or {}).items()
        }
        self.scan_table_path = "" if scan_table_path is None else str(scan_table_path)
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

    def validate(
        self,
        *,
        x_ns: float | None = None,
        y_ns: float | None = None,
        time_step_ns: float | None = None,
        variables: Mapping[str, float] | None = None,
    ) -> "PulseTableState":
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        x_ns = quantized_time_ns(self.x_ns if x_ns is None else x_ns, time_step_ns=step_ns, name="x_ns", allow_zero=True)
        y_ns = quantized_time_ns(self.y_ns if y_ns is None else y_ns, time_step_ns=step_ns, name="y_ns", allow_zero=True, allow_negative=True)
        variables_for_eval = self.scan_variable_values(x_ns=x_ns, y_ns=y_ns)
        variables_for_eval.update(dict(variables or {}))
        if not self.channels:
            raise ValueError("pulse table must have at least one channel.")
        if self.scan_points:
            raise ValueError("legacy x/y scan_points are no longer supported; link a named scan table file.")
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
        for key, variable in self.scan_bindings.items():
            normalize_scan_binding_key(key)
            normalize_scan_parameter_name(variable)
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
            period.duration_steps(x_ns=x_ns, y_ns=y_ns, time_step_ns=step_ns, variables=variables_for_eval)
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
                value_int = scan_numeric_value(value, variables=variables_for_eval, name=f"analog bus {bus_name!r} period {index} value")
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
            self.delay_steps(channel, x_ns=x_ns, y_ns=y_ns, time_step_ns=step_ns, variables=variables_for_eval)
        return self

    def with_x(self, x_ns: float) -> "PulseTableState":
        """Legacy x/y scans are no longer supported."""

        raise ValueError("legacy x/y scan variables are no longer supported; use with_scan_variables(name=value).")

    def with_xy(self, x_ns: float, y_ns: float = 0.0) -> "PulseTableState":
        """Legacy x/y scans are no longer supported."""

        raise ValueError("legacy x/y scan variables are no longer supported; use with_scan_variables(name=value).")

    def with_scan_points(self, scan_points: Sequence[Sequence[float] | float]) -> "PulseTableState":
        """Legacy x/y point-array scans are no longer supported."""

        raise ValueError("legacy x/y scan_points are no longer supported; link a named scan table file.")

    def with_scan_table_path(self, path: str | Path | None) -> "PulseTableState":
        """Return a copy linked to a named-parameter scan table file."""

        payload = self.to_dict()
        payload["scan_table_path"] = "" if path is None else str(path)
        if path:
            payload["scan_points"] = []
        return type(self).from_dict(payload)

    def with_scan_variables(self, **variables: float) -> "PulseTableState":
        """Return a copy with updated default named scan-variable values."""

        payload = self.to_dict()
        merged = dict(payload.get("scan_variables", {}))
        for name, value in variables.items():
            normalized = normalize_scan_parameter_name(name)
            merged[normalized] = float(value)
        payload["scan_variables"] = merged
        return type(self).from_dict(payload)

    def scan_variable_values(self, *, x_ns: float | None = None, y_ns: float | None = None) -> dict[str, float]:
        """Return default expression variables for named scan parameters."""

        return {str(name): float(value) for name, value in self.scan_variables.items()}

    def active_scan_parameters(self) -> list[str]:
        """Return named scan variables referenced by durations, delays, buses, or bindings."""

        active: set[str] = set()
        active.update(self.scan_bindings.values())
        for period in self.periods:
            active.update(scan_parameter_names_from_expr(period.duration))
        for value in self.delays.values():
            active.update(scan_parameter_names_from_expr(value))
        for entries in self.analog_bus_modes.values():
            for entry in entries:
                value = dict(entry).get("value")
                active.update(scan_parameter_names_from_expr(value))
        return sorted(active)

    def scan_table(self, *, time_step_ns: float | None = None) -> ScanParameterTable:
        """Load the linked named-parameter scan file."""

        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        required = list(self.active_scan_parameters())
        if self.scan_table_path:
            return load_scan_parameter_table(self.scan_table_path, time_step_ns=step_ns, required=required)
        if self.scan_points:
            raise ValueError("legacy x/y scan_points are no longer supported; link a named scan table file.")
        return ScanParameterTable((), tuple(), {})

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
                    normalized.append({"mode": mode, "value": None if mode == "hold" else value})
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
            x_ns=self.x_ns,
            y_ns=self.y_ns,
            scan_points=self.scan_points,
            scan_variables=self.scan_variables,
            scan_bindings=self.scan_bindings,
            scan_table_path=self.scan_table_path,
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
        """Channels that go high when an internal finite bracket restarts the table.

        With ``repeat_forever=True``, an internal finite repeat bracket is a
        nested loop.  After that loop count is exhausted, the FPGA finishes the
        post-bracket periods and then restarts the whole uploaded table.  If
        period 0 has high outputs, those channels will pulse once at that table
        boundary.  This is often what an oscilloscope shows as a slow periodic
        spike.
        """

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
        """Set one bus period mode and optional 10-bit value."""

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
            plan[period_index] = {"mode": mode, "value": value}
        self.analog_bus_modes[bus_name] = plan
        if validate:
            self.apply_analog_bus_modes_to_period_states()
            self.validate()
        return self

    def apply_analog_bus_modes_to_period_states(self) -> "PulseTableState":
        """Project logical bus mode/value rows back to underlying TTL bits."""

        variables = self.scan_variable_values()
        starts = self._period_start_steps(x_ns=self.x_ns, y_ns=self.y_ns, time_step_ns=self.time_step_ns, variables=variables)
        groups = self.bus_channels(min_width=1)
        for bus_name, members in groups.items():
            plan = self.analog_bus_plan(bus_name)
            for period_index, period in enumerate(self.periods):
                value = _analog_bus_value_at_tick(plan, starts, starts[period_index], variables=variables)
                states = list(period.states)
                for bit, channel in enumerate(members):
                    states[self.channel_index(channel)] = 1 if (int(value) >> bit) & 1 else 0
                self.periods[period_index] = PulsePeriod(period.duration, tuple(states), unit=period.unit, name=period.name)
        return self

    def analog_bus_value_at_period_start(self, period_index: int, bus_name: str) -> int:
        variables = self.scan_variable_values()
        starts = self._period_start_steps(x_ns=self.x_ns, y_ns=self.y_ns, time_step_ns=self.time_step_ns, variables=variables)
        return _analog_bus_value_at_tick(self.analog_bus_plan(bus_name), starts, starts[int(period_index)], variables=variables)

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

    def delay_steps(
        self,
        channel: str,
        *,
        x_ns: float | None = None,
        y_ns: float | None = None,
        time_step_ns: float | None = None,
        variables: Mapping[str, float] | None = None,
    ) -> int:
        raw = self.delays.get(channel, 0.0)
        unit = self.delay_units.get(channel, "ns")
        if unit not in UNITS_TO_NS:
            raise ValueError(f"unsupported delay unit {unit!r}.")
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        x_value = self.x_ns if x_ns is None else x_ns
        y_value = self.y_ns if y_ns is None else y_ns
        expression_vars = self.scan_variable_values(x_ns=x_value, y_ns=y_value)
        expression_vars.update(dict(variables or {}))
        return quantized_time_steps(
            eval_time_expr(raw, x_ns=x_value, y_ns=y_value, variables=expression_vars) * UNITS_TO_NS[unit],
            time_step_ns=step_ns,
            name=f"delay for {channel!r}",
            allow_zero=True,
            allow_negative=True,
        )

    def delay_ns(
        self,
        channel: str,
        *,
        x_ns: float | None = None,
        y_ns: float | None = None,
        time_step_ns: float | None = None,
        variables: Mapping[str, float] | None = None,
    ) -> float:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        return self.delay_steps(channel, x_ns=x_ns, y_ns=y_ns, time_step_ns=step_ns, variables=variables) * step_ns

    def total_duration_steps(
        self,
        *,
        x_ns: float | None = None,
        y_ns: float | None = None,
        time_step_ns: float | None = None,
        variables: Mapping[str, float] | None = None,
    ) -> int:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        x_ns = quantized_time_ns(self.x_ns if x_ns is None else x_ns, time_step_ns=step_ns, name="x_ns", allow_zero=True)
        y_ns = quantized_time_ns(self.y_ns if y_ns is None else y_ns, time_step_ns=step_ns, name="y_ns", allow_zero=True, allow_negative=True)
        expression_vars = self.scan_variable_values(x_ns=x_ns, y_ns=y_ns)
        expression_vars.update(dict(variables or {}))
        return sum(period.duration_steps(x_ns=x_ns, y_ns=y_ns, time_step_ns=step_ns, variables=expression_vars) for period in self.expanded_periods())

    def total_duration_ns(
        self,
        *,
        x_ns: float | None = None,
        y_ns: float | None = None,
        time_step_ns: float | None = None,
        variables: Mapping[str, float] | None = None,
    ) -> float:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        return self.total_duration_steps(x_ns=x_ns, y_ns=y_ns, time_step_ns=step_ns, variables=variables) * step_ns

    def _period_start_steps(
        self,
        *,
        x_ns: float,
        y_ns: float = 0.0,
        time_step_ns: float,
        variables: Mapping[str, float] | None = None,
    ) -> list[int]:
        starts = [0]
        expression_vars = self.scan_variable_values(x_ns=x_ns, y_ns=y_ns)
        expression_vars.update(dict(variables or {}))
        for period in self.periods:
            starts.append(starts[-1] + period.duration_steps(x_ns=x_ns, y_ns=y_ns, time_step_ns=time_step_ns, variables=expression_vars))
        return starts

    def to_sequence(
        self,
        *,
        name: str | None = None,
        x_ns: float | None = None,
        y_ns: float | None = None,
        time_step_ns: float | None = None,
        variables: Mapping[str, float] | None = None,
        expand_repeat: bool = True,
    ) -> PulseSequence:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        x_ns = self.x_ns if x_ns is None else quantized_time_ns(x_ns, time_step_ns=step_ns, name="x_ns", allow_zero=True)
        y_ns = self.y_ns if y_ns is None else quantized_time_ns(y_ns, time_step_ns=step_ns, name="y_ns", allow_zero=True, allow_negative=True)
        self.validate(x_ns=x_ns, y_ns=y_ns, time_step_ns=step_ns)
        expression_vars = self.scan_variable_values(x_ns=x_ns, y_ns=y_ns)
        expression_vars.update(dict(variables or {}))
        sequence = PulseSequence(name=name or self.name)
        starts: dict[str, int | None] = {channel: None for channel in self.channels}
        t_steps = 0
        periods = self.expanded_periods() if expand_repeat else list(self.periods)
        for period in periods:
            duration_steps = period.duration_steps(x_ns=x_ns, y_ns=y_ns, time_step_ns=step_ns, variables=expression_vars)
            next_t_steps = t_steps + duration_steps
            for channel, state in zip(self.channels, period.states):
                active_start = starts[channel]
                if state and active_start is None:
                    starts[channel] = t_steps
                elif not state and active_start is not None:
                    sequence = sequence.pulse(channel, active_start * step_ns * 1e-9, (t_steps - active_start) * step_ns * 1e-9)
                    starts[channel] = None
            t_steps = next_t_steps
        for channel, active_start in starts.items():
            if active_start is not None:
                sequence = sequence.pulse(channel, active_start * step_ns * 1e-9, (t_steps - active_start) * step_ns * 1e-9)
        for channel in self.channels:
            delay_steps = self.delay_steps(channel, x_ns=x_ns, y_ns=y_ns, time_step_ns=step_ns, variables=expression_vars)
            if delay_steps:
                sequence = sequence.delay(channel, delay_steps * step_ns * 1e-9)
        return sequence

    def compile(
        self,
        *,
        clock_hz: float,
        trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
        x_ns: float | None = None,
        y_ns: float | None = None,
        variables: Mapping[str, float] | None = None,
        repeat_forever: bool | None = None,
    ):
        from ..devices.sequencer import compile_pulse_table_runtime_program

        clock_hz = positive_float(clock_hz, "clock_hz")
        return compile_pulse_table_runtime_program(
            self,
            channels=self.channels,
            clock_hz=clock_hz,
            trigger_channels=trigger_channels,
            x_ns=x_ns,
            y_ns=y_ns,
            variables=variables,
            repeat_forever=self.repeat_forever if repeat_forever is None else bool(repeat_forever),
        )

    def compile_scan(
        self,
        *,
        clock_hz: float,
        scan_points: Sequence[Sequence[float] | float] | None = None,
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
            scan_points=self.scan_points if scan_points is None else scan_points,
            repeat_forever=self.repeat_forever if repeat_forever is None else bool(repeat_forever),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": 2,
            "name": self.name,
            "channels": list(self.channels),
            "x_ns": self.x_ns,
            "y_ns": self.y_ns,
            "scan_points": [list(point) for point in self.scan_points],
            "scan_variables": dict(self.scan_variables),
            "scan_bindings": dict(self.scan_bindings),
            "scan_table_path": self.scan_table_path,
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PulseTableState":
        if payload.get("schema", cls.schema) != cls.schema:
            raise ValueError("unsupported pulse table schema.")
        return cls(
            name=str(payload["name"]) if "name" in payload else None,
            channels=payload["channels"],
            x_ns=float(payload.get("x_ns", 0.0)),
            y_ns=float(payload.get("y_ns", 0.0)),
            scan_points=payload.get("scan_points", ()),
            scan_variables=dict(payload.get("scan_variables", {})),
            scan_bindings=dict(payload.get("scan_bindings", {})),
            scan_table_path=payload.get("scan_table_path", ""),
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
    """Infer logical buses from labels such as ``da_dipole[0]`` ... ``[9]``.

    The returned member list is ordered least-significant bit first by the
    bracket index, while each member remains a real FPGA/TTL channel.
    """

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


def _analog_bus_value_at_tick(
    plan: Sequence[Mapping[str, object]],
    starts: Sequence[int],
    tick: int,
    *,
    variables: Mapping[str, float] | None = None,
) -> int:
    tick = int(tick)
    anchors: list[tuple[int, int, str, int]] = []
    for index, entry in enumerate(plan):
        mode = str(entry.get("mode", "hold")).lower()
        if mode not in {"edge", "ramp"}:
            continue
        value = entry.get("value")
        if value is None:
            continue
        anchors.append((index, int(starts[index]), mode, scan_numeric_value(value, variables=variables, name="analog bus value")))
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


def normalize_scan_parameter_name(name: object) -> str:
    """Return a valid host-side scan parameter name."""

    text = str(name or "").strip()
    if not SCAN_NAME_RE.fullmatch(text):
        raise ValueError(f"scan parameter name {text!r} must look like a Python identifier.")
    return text


def normalize_scan_binding_key(key: object) -> str:
    text = str(key or "").strip()
    if not text:
        raise ValueError("scan binding key must not be empty.")
    return text


def scan_parameter_names_from_expr(value: object) -> set[str]:
    """Return variable names referenced by a numeric/time expression."""

    if isinstance(value, Number) or value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    try:
        tree = ast.parse(_insert_mul_before_vars(text), mode="eval")
    except Exception:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(normalize_scan_parameter_name(node.id))
    return names


def scan_numeric_value(value: object, *, variables: Mapping[str, float] | None = None, name: str = "scan value") -> int:
    """Evaluate a scalar scan expression and return the nearest integer."""

    out = eval_time_expr(value, variables=variables)
    rounded = int(round(out))
    if not math.isclose(out, rounded, rel_tol=GRID_RTOL, abs_tol=GRID_ATOL_STEPS):
        raise ValueError(f"{name}={out:g} must be an integer.")
    return rounded


def load_scan_parameter_table(
    path: str | Path,
    *,
    time_step_ns: float = 1.0,
    required: Sequence[str] | None = None,
) -> ScanParameterTable:
    """Load an ordered named-parameter scan table.

    Text files may declare variables with a comment header, for example
    ``# vars: tau(us), delay(ns), amp``.  Without that comment, the first row
    may be a normal CSV/whitespace header.  Units ``ns/us/ms/s`` are converted
    to ns so timing expressions stay unitless on the FPGA/host boundary.
    """

    table_path = Path(path)
    required_names = [normalize_scan_parameter_name(name) for name in (required or [])]
    if not table_path.exists():
        raise FileNotFoundError(f"scan table file not found: {table_path}")
    if table_path.suffix.lower() == ".json":
        return _load_json_scan_table(table_path, time_step_ns=time_step_ns, required=required_names)
    return _load_text_scan_table(table_path, time_step_ns=time_step_ns, required=required_names)


def _load_json_scan_table(path: Path, *, time_step_ns: float, required: Sequence[str]) -> ScanParameterTable:
    payload = json.loads(path.read_text(encoding="utf-8"))
    units: dict[str, str] = {}
    if isinstance(payload, Mapping):
        units = {normalize_scan_parameter_name(k): str(v).strip() for k, v in dict(payload.get("units", {})).items()}
        raw_rows = payload.get("points", payload.get("rows", []))
    else:
        raw_rows = payload
    if not isinstance(raw_rows, Sequence):
        raise ValueError("JSON scan table must be a list of row objects or a {'points': [...]} mapping.")
    rows: list[dict[str, float]] = []
    names: list[str] = []
    for row_index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"JSON scan row {row_index} must be an object with named columns.")
        normalized: dict[str, float] = {}
        for name, value in row.items():
            scan_name = normalize_scan_parameter_name(name)
            if scan_name not in names:
                names.append(scan_name)
            normalized[scan_name] = _scan_file_value(value, units.get(scan_name, ""), time_step_ns=time_step_ns, name=f"{scan_name} row {row_index}")
        rows.append(normalized)
    _validate_scan_table_columns(names, rows, required)
    return ScanParameterTable(tuple(names), tuple(rows), units, str(path))


def _load_text_scan_table(path: Path, *, time_step_ns: float, required: Sequence[str]) -> ScanParameterTable:
    names: list[str] = []
    units: dict[str, str] = {}
    data_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header_match = SCAN_HEADER_RE.match(line)
        if header_match:
            names, units = _parse_scan_column_specs(header_match.group("vars"))
            continue
        if line.startswith("#"):
            continue
        data_lines.append(line)
    if not data_lines:
        raise ValueError(f"scan table {path} has no data rows.")
    first_tokens = _split_scan_line(data_lines[0])
    if not names and first_tokens and any(re.search(r"[A-Za-z_]", token) for token in first_tokens):
        names, units = _parse_scan_column_specs(data_lines.pop(0))
    if not names:
        if len(required) == len(first_tokens) and required:
            names = list(required)
        elif len(first_tokens) == 1 and len(required) == 1:
            names = [required[0]]
        else:
            raise ValueError("scan table needs '# vars: ...' or a header row when more than one column is present.")
    rows: list[dict[str, float]] = []
    for row_index, line in enumerate(data_lines):
        values = _split_scan_line(line)
        if len(values) != len(names):
            raise ValueError(f"scan table row {row_index} has {len(values)} values but {len(names)} columns.")
        row = {
            name: _scan_file_value(value, units.get(name, ""), time_step_ns=time_step_ns, name=f"{name} row {row_index}")
            for name, value in zip(names, values)
        }
        rows.append(row)
    _validate_scan_table_columns(names, rows, required)
    return ScanParameterTable(tuple(names), tuple(rows), units, str(path))


def _parse_scan_column_specs(text: str) -> tuple[list[str], dict[str, str]]:
    names: list[str] = []
    units: dict[str, str] = {}
    for token in _split_scan_line(text):
        match = SCAN_TOKEN_RE.fullmatch(token.strip())
        if not match:
            raise ValueError(f"bad scan column spec {token!r}; use name or name(unit).")
        name = normalize_scan_parameter_name(match.group("name"))
        if name in names:
            raise ValueError(f"scan table column {name!r} is duplicated.")
        unit = str(match.group("unit") or match.group("bracket_unit") or "").strip()
        names.append(name)
        if unit:
            units[name] = unit
    return names, units


def _split_scan_line(text: str) -> list[str]:
    stripped = str(text).strip()
    if "," in stripped:
        return [token.strip() for token in stripped.split(",") if token.strip()]
    return [token.strip() for token in stripped.split() if token.strip()]


def _scan_file_value(value: object, unit: str, *, time_step_ns: float, name: str) -> float:
    numeric = float(value)
    unit = str(unit or "").strip()
    if unit:
        unit_key = unit.lower()
        if unit_key not in TIME_SCAN_UNITS:
            raise ValueError(f"{name} uses unsupported scan unit {unit!r}.")
        return quantized_time_ns(numeric * TIME_SCAN_UNITS[unit_key], time_step_ns=time_step_ns, name=name, allow_zero=True, allow_negative=True)
    return numeric


def _validate_scan_table_columns(names: Sequence[str], rows: Sequence[Mapping[str, float]], required: Sequence[str]) -> None:
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(
            f"scan table is missing required column(s): {missing}. "
            f"Available columns are {list(names)}. Make the '# vars:' header match the GUI Params row."
        )
    for row_index, row in enumerate(rows):
        missing_row = [name for name in names if name not in row]
        if missing_row:
            raise ValueError(f"scan table row {row_index} is missing column(s): {missing_row}.")


def eval_time_expr(
    value: float | str,
    *,
    x_ns: float = 0.0,
    y_ns: float = 0.0,
    variables: Mapping[str, float] | None = None,
) -> float:
    """Evaluate a small numeric expression with host-side scan variables."""

    if isinstance(value, Number):
        out = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("time expression must not be empty.")
        expression_vars = {normalize_scan_parameter_name(k): float(v) for k, v in dict(variables or {}).items()}
        out = _SafeEval(expression_vars).eval(text)
    if not math.isfinite(out):
        raise ValueError("time expression must be finite.")
    return out


def positive_time_step_ns(value: float | str) -> float:
    out = eval_time_expr(value, x_ns=0.0, y_ns=0.0)
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
    value = eval_time_expr(value_ns, x_ns=0.0, y_ns=0.0)
    step = positive_time_step_ns(time_step_ns)
    raw_steps = value / step
    steps = int(round(raw_steps))
    if not math.isclose(raw_steps, steps, rel_tol=GRID_RTOL, abs_tol=GRID_ATOL_STEPS):
        raise ValueError(f"{name}={value:g} ns is not an integer multiple of time_step_ns={step:g} ns.")
    if steps < 0 and not allow_negative:
        raise ValueError(f"{name} must be >= 0 ns.")
    if steps == 0 and not allow_zero:
        raise ValueError(f"{name} must be at least one time step.")
    return steps


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


def affine_time_expr(value: float | str, *, unit: str = "ns", time_step_ns: float = 1.0, coeff_frac_bits: int = 8) -> tuple[int, int, int]:
    """Return ``(base_ticks, x_coeff_fixed, y_coeff_fixed)`` for scan timing."""

    if unit not in UNITS_TO_NS:
        raise ValueError(f"unsupported time unit {unit!r}.")
    base, x_coeff, y_coeff = _SafeEval(0.0, 0.0).affine(value)
    unit_scale = UNITS_TO_NS[unit]
    step_ns = positive_time_step_ns(time_step_ns)
    base_ticks_raw = base * unit_scale / step_ns
    base_ticks = int(round(base_ticks_raw))
    if not math.isclose(base_ticks_raw, base_ticks, rel_tol=GRID_RTOL, abs_tol=GRID_ATOL_STEPS):
        raise ValueError(f"affine base {base * unit_scale:g} ns is not an integer multiple of time_step_ns={step_ns:g} ns.")
    scale = 1 << int(coeff_frac_bits)
    x_fixed = int(round(x_coeff * unit_scale * scale))
    y_fixed = int(round(y_coeff * unit_scale * scale))
    if not math.isclose(x_fixed / scale, x_coeff * unit_scale, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"x coefficient {x_coeff:g} cannot be represented with {coeff_frac_bits} fractional bits.")
    if not math.isclose(y_fixed / scale, y_coeff * unit_scale, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"y coefficient {y_coeff:g} cannot be represented with {coeff_frac_bits} fractional bits.")
    return base_ticks, x_fixed, y_fixed


def affine_named_time_expr(
    value: float | str,
    *,
    variable_names: Sequence[str],
    unit: str = "ns",
    time_step_ns: float = 1.0,
    coeff_frac_bits: int = 8,
) -> tuple[int, dict[str, int]]:
    """Return base ticks and fixed-point coefficients for named variables."""

    if unit not in UNITS_TO_NS:
        raise ValueError(f"unsupported time unit {unit!r}.")
    axes = [normalize_scan_parameter_name(name) for name in variable_names]
    base, coeffs = _affine_named_expr(value, axes)
    unit_scale = UNITS_TO_NS[unit]
    step_ns = positive_time_step_ns(time_step_ns)
    base_ticks_raw = base * unit_scale / step_ns
    base_ticks = int(round(base_ticks_raw))
    if not math.isclose(base_ticks_raw, base_ticks, rel_tol=GRID_RTOL, abs_tol=GRID_ATOL_STEPS):
        raise ValueError(f"affine base {base * unit_scale:g} ns is not an integer multiple of time_step_ns={step_ns:g} ns.")
    scale = 1 << int(coeff_frac_bits)
    fixed: dict[str, int] = {}
    for name in axes:
        coeff = coeffs.get(name, 0.0)
        fixed_value = int(round(coeff * unit_scale * scale))
        if not math.isclose(fixed_value / scale, coeff * unit_scale, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{name} coefficient {coeff:g} cannot be represented with {coeff_frac_bits} fractional bits.")
        fixed[name] = fixed_value
    unknown = sorted(name for name in coeffs if name not in set(axes) and abs(coeffs[name]) > 0.0)
    if unknown:
        raise ValueError(f"hardware scan expression uses variable(s) not present in the scan table: {unknown}.")
    return base_ticks, fixed


def _affine_named_expr(value: float | str, variable_names: Sequence[str]) -> tuple[float, dict[str, float]]:
    axes = set(variable_names)
    if isinstance(value, Number):
        return float(value), {name: 0.0 for name in axes}
    text = str(value).strip()
    if not text:
        raise ValueError("time expression must not be empty.")
    base, coeffs = _affine_named_visit(ast.parse(_insert_mul_before_vars(text), mode="eval").body, axes)
    return base, coeffs


def _affine_named_visit(node, variable_names: set[str]) -> tuple[float, dict[str, float]]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value), {name: 0.0 for name in variable_names}
    if isinstance(node, ast.Name):
        name = normalize_scan_parameter_name(node.id)
        if name not in variable_names:
            return 0.0, {name: 1.0}
        return 0.0, {axis: 1.0 if axis == name else 0.0 for axis in variable_names}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        base, coeffs = _affine_named_visit(node.operand, variable_names)
        if isinstance(node.op, ast.USub):
            return -base, {name: -value for name, value in coeffs.items()}
        return base, coeffs
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left_base, left_coeffs = _affine_named_visit(node.left, variable_names)
        right_base, right_coeffs = _affine_named_visit(node.right, variable_names)
        sign = -1.0 if isinstance(node.op, ast.Sub) else 1.0
        names = set(left_coeffs) | set(right_coeffs)
        return left_base + sign * right_base, {name: left_coeffs.get(name, 0.0) + sign * right_coeffs.get(name, 0.0) for name in names}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_base, left_coeffs = _affine_named_visit(node.left, variable_names)
        right_base, right_coeffs = _affine_named_visit(node.right, variable_names)
        left_has_var = any(abs(value) > 0.0 for value in left_coeffs.values())
        right_has_var = any(abs(value) > 0.0 for value in right_coeffs.values())
        if left_has_var and right_has_var:
            raise ValueError("hardware scan timing only supports affine expressions; products of variables are not supported.")
        if left_has_var:
            return left_base * right_base, {name: value * right_base for name, value in left_coeffs.items()}
        if right_has_var:
            return left_base * right_base, {name: value * left_base for name, value in right_coeffs.items()}
        return left_base * right_base, {name: 0.0 for name in variable_names}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left_base, left_coeffs = _affine_named_visit(node.left, variable_names)
        right_base, right_coeffs = _affine_named_visit(node.right, variable_names)
        if any(abs(value) > 0.0 for value in right_coeffs.values()) or right_base == 0.0:
            raise ValueError("hardware scan timing only supports division by a nonzero constant.")
        return left_base / right_base, {name: value / right_base for name, value in left_coeffs.items()}
    raise ValueError("hardware scan timing only supports affine expressions in scan variables.")


class _SafeEval:
    _binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }
    _unary = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}

    def __init__(
        self,
        variables: Mapping[str, float] | None = None,
        y_ns: float | None = None,
    ):
        if y_ns is not None:
            self.variables = {"x": float(variables or 0.0), "y": float(y_ns)}
        else:
            self.variables = {normalize_scan_parameter_name(k): float(v) for k, v in dict(variables or {}).items()}

    def eval(self, text: str) -> float:
        return float(self._visit(ast.parse(_insert_mul_before_vars(text), mode="eval").body))

    def _visit(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and normalize_scan_parameter_name(node.id) in self.variables:
            return self.variables[normalize_scan_parameter_name(node.id)]
        if isinstance(node, ast.BinOp) and type(node.op) in self._binops:
            return self._binops[type(node.op)](self._visit(node.left), self._visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary:
            return self._unary[type(node.op)](self._visit(node.operand))
        raise ValueError("time expression may only use numbers, scan variables, +, -, *, /, **, and parentheses.")

    def affine(self, value: float | str) -> tuple[float, float, float]:
        if isinstance(value, Number):
            return float(value), 0.0, 0.0
        text = str(value).strip()
        if not text:
            raise ValueError("time expression must not be empty.")
        return self._affine_visit(ast.parse(_insert_mul_before_vars(text), mode="eval").body)

    def _affine_visit(self, node) -> tuple[float, float, float]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value), 0.0, 0.0
        if isinstance(node, ast.Name) and node.id == "x":
            return 0.0, 1.0, 0.0
        if isinstance(node, ast.Name) and node.id == "y":
            return 0.0, 0.0, 1.0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            base, x_coeff, y_coeff = self._affine_visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -base, -x_coeff, -y_coeff
            return base, x_coeff, y_coeff
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left = self._affine_visit(node.left)
            right = self._affine_visit(node.right)
            sign = -1.0 if isinstance(node.op, ast.Sub) else 1.0
            return left[0] + sign * right[0], left[1] + sign * right[1], left[2] + sign * right[2]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left = self._affine_visit(node.left)
            right = self._affine_visit(node.right)
            if left[1] == 0.0 and left[2] == 0.0:
                return right[0] * left[0], right[1] * left[0], right[2] * left[0]
            if right[1] == 0.0 and right[2] == 0.0:
                return left[0] * right[0], left[1] * right[0], left[2] * right[0]
            raise ValueError("hardware scan timing only supports affine x/y expressions; products of variables are not supported.")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = self._affine_visit(node.left)
            right = self._affine_visit(node.right)
            if right[1] != 0.0 or right[2] != 0.0 or right[0] == 0.0:
                raise ValueError("hardware scan timing only supports division by a nonzero constant.")
            return left[0] / right[0], left[1] / right[0], left[2] / right[0]
        raise ValueError("hardware scan timing only supports affine expressions in x and y.")


def _insert_mul_before_vars(text: str) -> str:
    normalized = str(text)
    out: list[str] = []
    for index, char in enumerate(normalized):
        if char.isalpha() or char == "_":
            prev = normalized[index - 1] if index > 0 else ""
            if prev.isdigit() or prev == ".":
                start = index - 1
                while start >= 0 and (normalized[start].isdigit() or normalized[start] == "."):
                    start -= 1
                if start < 0 or not (normalized[start].isalnum() or normalized[start] == "_"):
                    out.append("*")
        out.append(char)
    return "".join(out)


def _insert_mul_before_x(text: str) -> str:
    return _insert_mul_before_vars(text)


__all__ = [
    "ANALOG_BUS_MODES",
    "PulsePeriod",
    "PulseTableState",
    "ScanParameterTable",
    "default_pulse_name",
    "default_periods",
    "default_visible_channels",
    "affine_named_time_expr",
    "eval_time_expr",
    "infer_bus_channels",
    "load_scan_parameter_table",
    "normalize_scan_parameter_name",
    "positive_time_step_ns",
    "quantized_time_ns",
    "quantized_time_steps",
    "scan_numeric_value",
    "scan_parameter_names_from_expr",
]
