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
from typing import Iterable, Mapping, Sequence

from .sequence import PulseSequence, channel_names, positive_float


UNITS_TO_NS = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0, "str (ns)": 1.0}
GRID_RTOL = 1e-12
GRID_ATOL_STEPS = 1e-9


def default_pulse_name() -> str:
    return "pulse_" + datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class PulsePeriod:
    """One period-card in the pulse GUI."""

    duration: float | str
    states: tuple[int, ...]
    unit: str = "ns"
    name: str = ""

    def duration_steps(self, *, x_ns: float = 0.0, time_step_ns: float = 1.0) -> int:
        value = eval_time_expr(self.duration, x_ns=x_ns)
        unit = str(self.unit or "ns")
        if unit not in UNITS_TO_NS:
            raise ValueError(f"unsupported pulse duration unit {unit!r}.")
        return quantized_time_steps(value * UNITS_TO_NS[unit], time_step_ns=time_step_ns, name="period duration", allow_zero=False)

    def duration_ns(self, *, x_ns: float = 0.0, time_step_ns: float | None = None) -> float:
        value = eval_time_expr(self.duration, x_ns=x_ns)
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
        time_step_ns: float = 1.0,
        repeat_start: int | None = None,
        repeat_end: int | None = None,
        repeat_count: int = 1,
        visible_channels: Sequence[str] | None = None,
        channel_labels: Mapping[str, str] | None = None,
    ):
        self.channels = list(channel_names(channels, "channels"))
        self.name = str(name) if name is not None else default_pulse_name()
        self.time_step_ns = positive_time_step_ns(time_step_ns)
        self.x_ns = quantized_time_ns(x_ns, time_step_ns=self.time_step_ns, name="x_ns", allow_zero=True)
        self.periods = list(periods or default_periods(self.channels))
        self.delays = {str(k): v for k, v in dict(delays or {}).items()}
        self.delay_units = {str(k): str(v) for k, v in dict(delay_units or {}).items()}
        self.repeat_start = None if repeat_start is None else int(repeat_start)
        self.repeat_end = None if repeat_end is None else int(repeat_end)
        self.repeat_count = int(repeat_count)
        self.visible_channels = list(channel_names(visible_channels, "visible_channels", allow_empty=True)) if visible_channels is not None else default_visible_channels(self.channels)
        self.channel_labels = {str(k): str(v) for k, v in dict(channel_labels or {}).items()}
        self.validate()

    def validate(self, *, x_ns: float | None = None, time_step_ns: float | None = None) -> "PulseTableState":
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        x_ns = quantized_time_ns(self.x_ns if x_ns is None else x_ns, time_step_ns=step_ns, name="x_ns", allow_zero=True)
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
        width = len(self.channels)
        if not self.periods:
            raise ValueError("pulse table must have at least one period.")
        for index, period in enumerate(self.periods):
            if len(period.states) != width:
                raise ValueError(f"period {index} has {len(period.states)} states but {width} channels.")
            for value in period.states:
                if int(value) not in (0, 1):
                    raise ValueError("period states must be 0 or 1.")
            period.duration_steps(x_ns=x_ns, time_step_ns=step_ns)
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
            self.delay_steps(channel, x_ns=x_ns, time_step_ns=step_ns)
        return self

    def with_x(self, x_ns: float) -> "PulseTableState":
        """Return a copy of this editable table with a different ``x`` value."""

        payload = self.to_dict()
        payload["x_ns"] = float(x_ns)
        return type(self).from_dict(payload)

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
            time_step_ns=self.time_step_ns,
            repeat_start=self.repeat_start,
            repeat_end=self.repeat_end,
            repeat_count=self.repeat_count,
            visible_channels=visible,
            channel_labels={channel: value for channel, value in self.channel_labels.items() if channel in channels},
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

    def delay_steps(self, channel: str, *, x_ns: float | None = None, time_step_ns: float | None = None) -> int:
        raw = self.delays.get(channel, 0.0)
        unit = self.delay_units.get(channel, "ns")
        if unit not in UNITS_TO_NS:
            raise ValueError(f"unsupported delay unit {unit!r}.")
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        x_value = self.x_ns if x_ns is None else x_ns
        return quantized_time_steps(
            eval_time_expr(raw, x_ns=x_value) * UNITS_TO_NS[unit],
            time_step_ns=step_ns,
            name=f"delay for {channel!r}",
            allow_zero=True,
            allow_negative=True,
        )

    def delay_ns(self, channel: str, *, x_ns: float | None = None, time_step_ns: float | None = None) -> float:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        return self.delay_steps(channel, x_ns=x_ns, time_step_ns=step_ns) * step_ns

    def total_duration_steps(self, *, x_ns: float | None = None, time_step_ns: float | None = None) -> int:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        x_ns = quantized_time_ns(self.x_ns if x_ns is None else x_ns, time_step_ns=step_ns, name="x_ns", allow_zero=True)
        return sum(period.duration_steps(x_ns=x_ns, time_step_ns=step_ns) for period in self.expanded_periods())

    def total_duration_ns(self, *, x_ns: float | None = None, time_step_ns: float | None = None) -> float:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        return self.total_duration_steps(x_ns=x_ns, time_step_ns=step_ns) * step_ns

    def to_sequence(
        self,
        *,
        name: str | None = None,
        x_ns: float | None = None,
        time_step_ns: float | None = None,
        expand_repeat: bool = True,
    ) -> PulseSequence:
        step_ns = self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)
        x_ns = self.x_ns if x_ns is None else quantized_time_ns(x_ns, time_step_ns=step_ns, name="x_ns", allow_zero=True)
        self.validate(x_ns=x_ns, time_step_ns=step_ns)
        sequence = PulseSequence(name=name or self.name)
        starts: dict[str, int | None] = {channel: None for channel in self.channels}
        t_steps = 0
        periods = self.expanded_periods() if expand_repeat else list(self.periods)
        for period in periods:
            duration_steps = period.duration_steps(x_ns=x_ns, time_step_ns=step_ns)
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
            delay_steps = self.delay_steps(channel, x_ns=x_ns, time_step_ns=step_ns)
            if delay_steps:
                sequence = sequence.delay(channel, delay_steps * step_ns * 1e-9)
        return sequence

    def compile(
        self,
        *,
        clock_hz: float,
        trigger_channels: Sequence[str] = ("qcm_trigger", "camera_trigger", "trig"),
        x_ns: float | None = None,
    ):
        from ..devices.sequencer import compile_pulse_table_runtime_program

        clock_hz = positive_float(clock_hz, "clock_hz")
        return compile_pulse_table_runtime_program(
            self,
            channels=self.channels,
            clock_hz=clock_hz,
            trigger_channels=trigger_channels,
            x_ns=x_ns,
            repeat_forever=True,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": 1,
            "name": self.name,
            "channels": list(self.channels),
            "x_ns": self.x_ns,
            "time_step_ns": self.time_step_ns,
            "periods": [period.to_dict() for period in self.periods],
            "visible_channels": list(self.visible_channels),
            "channel_labels": dict(self.channel_labels),
            "delays": dict(self.delays),
            "delay_units": dict(self.delay_units),
            "repeat_start": self.repeat_start,
            "repeat_end": self.repeat_end,
            "repeat_count": self.repeat_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PulseTableState":
        if payload.get("schema", cls.schema) != cls.schema:
            raise ValueError("unsupported pulse table schema.")
        return cls(
            name=str(payload["name"]) if "name" in payload else None,
            channels=payload["channels"],
            x_ns=float(payload.get("x_ns", 0.0)),
            time_step_ns=float(payload.get("time_step_ns", 1.0)),
            periods=[PulsePeriod.from_dict(item) for item in payload.get("periods", [])],
            visible_channels=payload.get("visible_channels"),
            channel_labels=dict(payload.get("channel_labels", {})),
            delays=dict(payload.get("delays", {})),
            delay_units=dict(payload.get("delay_units", {})),
            repeat_start=payload.get("repeat_start"),
            repeat_end=payload.get("repeat_end"),
            repeat_count=int(payload.get("repeat_count", 1)),
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
    def from_sequence(cls, sequence: PulseSequence, *, channels: Sequence[str], clock_hz: float = 100_000_000.0) -> "PulseTableState":
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
    preferred = [channel for channel in ("trap", "cooling", "probe", "qcm_trigger") if channel in channels]
    if preferred:
        return preferred
    return channels[: min(8, len(channels))]


def eval_time_expr(value: float | str, *, x_ns: float = 0.0) -> float:
    """Evaluate a small numeric expression with the variable ``x`` in ns."""

    if isinstance(value, Number):
        out = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("time expression must not be empty.")
        out = _SafeEval(float(x_ns)).eval(text)
    if not math.isfinite(out):
        raise ValueError("time expression must be finite.")
    return out


def positive_time_step_ns(value: float | str) -> float:
    out = eval_time_expr(value, x_ns=0.0)
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
    value = eval_time_expr(value_ns, x_ns=0.0)
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


class _SafeEval:
    _binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }
    _unary = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}

    def __init__(self, x_ns: float):
        self.x_ns = x_ns

    def eval(self, text: str) -> float:
        return float(self._visit(ast.parse(_insert_mul_before_x(text), mode="eval").body))

    def _visit(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "x":
            return self.x_ns
        if isinstance(node, ast.BinOp) and type(node.op) in self._binops:
            return self._binops[type(node.op)](self._visit(node.left), self._visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary:
            return self._unary[type(node.op)](self._visit(node.operand))
        raise ValueError("time expression may only use numbers, x, +, -, *, /, **, and parentheses.")


def _insert_mul_before_x(text: str) -> str:
    out = []
    previous = ""
    for char in text.replace("X", "x"):
        if char == "x" and (previous.isdigit() or previous == "."):
            out.append("*")
        out.append(char)
        if not char.isspace():
            previous = char
    return "".join(out)


__all__ = [
    "PulsePeriod",
    "PulseTableState",
    "default_pulse_name",
    "default_periods",
    "default_visible_channels",
    "eval_time_expr",
    "positive_time_step_ns",
    "quantized_time_ns",
    "quantized_time_steps",
]
