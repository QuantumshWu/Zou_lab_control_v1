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
import logging
import math
from numbers import Number
import re
from typing import Iterable, Mapping, Sequence

from .sequence import (CLOCK_GRID_ATOL_TICKS as GRID_ATOL_STEPS,
                       CLOCK_GRID_RTOL as GRID_RTOL, DEFAULT_CLOCK_HZ,
                       READOUT_GAP_SECONDS,
                       PulseSequence, channel_names, positive_float)


UNITS_TO_NS = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0, "str (ns)": 1.0}
BUS_LABEL_RE = re.compile(r"^(?P<base>.+)\[(?P<bit>\d+)\]$")
ANALOG_BUS_MODES = ("hold", "edge", "ramp")


def bus_zero_code(n_bits: int) -> int:
    """Mid-scale OFFSET-BINARY code of an n-bit DAC bus = the code for TRUE 0 V.

    The DAC driver is bipolar offset-binary: wire code 0 = negative full scale,
    code 2^(B-1) (=512 for the 10-bit buses) = 0 V, code 2^B-1 = +FS-1LSB.  A B-bit
    bus has 2^B codes -- an EVEN count, so a symmetric range around zero is
    impossible; the industry convention (and ours) puts true zero at 2^(B-1), making
    the SIGNED user range asymmetric by one LSB: -2^(B-1) .. +2^(B-1)-1."""
    return 1 << (max(1, int(n_bits)) - 1)


def bus_signed_range(n_bits: int) -> tuple[int, int]:
    """User-facing SIGNED value range of an n-bit bus: (-2^(B-1), +2^(B-1)-1).

    ALL user-layer DAC values (GUI fields, ``analog_bus_modes`` entries, scan-table
    dac columns, ``ScanSlot.nominal``) are SIGNED LSB counts around true zero; the
    wire layer (RuntimeBusSegment values, program scan_points, the RTL) carries the
    offset-binary code = signed + ``bus_zero_code``.  The conversion happens exactly
    once, in the compilers."""
    half = bus_zero_code(n_bits)
    return (-half, half - 1)

#: OUTPUT delay magnitude cap, in clock ticks.  A per-channel (TTL) OR per-bus (DAC) delay
#: ``d`` is realized as ``output[t] = undelayed[t-d]`` by a per-signal EVENT SCHEDULER; the
#: cap is the 32-bit delay field (~42.9 s at 20 ns), the SAME for channels and buses.  The
#: GUI clamps each |delay| <= this; the real working limit is toggles IN FLIGHT (the
#: event-FIFO depth), which the compiler checks and reports.
DELAY_MAX_TICKS = (1 << 31) - 1

@dataclass(frozen=True)
class FieldKind:
    """What ONE bindable pulse FIELD kind supports.

    A field can carry a SCAN slot (rewrite to a streamed variable ``sN``, values from
    the scan table) and/or an API slot (a NAMED HANDLE ``aN`` that KEEPS the field's
    concrete value -- a stable label an external caller / Task sets by name via
    ``state.set_api("a1", v)`` / ``state.a1 = v``, without parsing "which signal of which
    period").  ``target`` describes the field's target-token shape.  ONE row per kind --
    the single source the scan/API rules derive from, so delay's "API yes, SCAN no"
    (a per-channel/bus delay is a FIXED output delay line, not a sweepable value) is ONE
    fact here, not three docstrings + two ``__post_init__`` rejections + a bind guard."""

    api: bool
    scan: bool
    target: str   # the target-token shape: period | bus@period | channel

#: The SINGLE SOURCE OF TRUTH for bindable field kinds.  Order matters only for the
#: derived tuples below; membership is what the rest of the module checks.
FIELD_KINDS = {
    "duration": FieldKind(api=True, scan=True, target="period"),
    "delay":    FieldKind(api=True, scan=False, target="channel"),   # delay can API, never SCAN
    "dac":      FieldKind(api=True, scan=True, target="bus@period"),
}

#: Derived from FIELD_KINDS -- NEVER hand-maintained.  A scan slot may bind a kind whose
#: ``scan`` is True; an API slot any kind whose ``api`` is True (delay qualifies for API only).
SCAN_SLOT_KINDS = tuple(k for k, v in FIELD_KINDS.items() if v.scan)
API_SLOT_KINDS = tuple(k for k, v in FIELD_KINDS.items() if v.api)
SLOT_VAR_RE = re.compile(r"^s(?P<index>\d+)$")
API_VAR_RE = re.compile(r"^a(?P<index>\d+)$")


def api_var(index: int) -> str:
    """Return the API-slot handle name for 1-based ``index`` (``a1``, ``a2`` ...)."""

    if int(index) < 1:
        raise ValueError("api slot index must be >= 1.")
    return f"a{int(index)}"


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
    field: a period index (``duration``) or ``"<bus>@<period_index>"`` (``dac``).
    ``label`` is a short human name for GUI lists.  ``unit`` records the field's
    display unit; ``scan_table`` values are stored as the field's final physical
    quantity (ns for time slots, the integer DAC code for ``dac`` slots).

    A per-channel delay is a FIXED output delay (a delay line) and is NOT
    scannable, so ``"delay"`` is not a valid slot kind.
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
        return self.kind == "duration"

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
class ApiSlot:
    """One API-settable field handle.

    ``name`` is the handle (``a1``, ``a2`` ...); ``kind`` is one of
    :data:`API_SLOT_KINDS`; ``target`` identifies the bound field exactly like
    :class:`ScanSlot` (period index for ``duration``, channel name for ``delay``,
    ``"<bus>@<period_index>"`` for ``dac``); ``unit`` is the unit a bare
    ``set_api(name, value)`` interprets ``value`` in.  Every slot's ``name`` is UNIQUE
    within a :class:`PulseTableState` -- one handle binds exactly one field, just like
    the GUI's per-cell dot allocates a fresh ``a<N>`` each time -- so the data layer and
    the GUI agree on what a name can mean.  Unlike a scan slot, the bound field keeps
    its concrete value -- the slot is only a label.
    """

    name: str
    kind: str
    target: str
    unit: str = "ns"

    def __post_init__(self) -> None:
        if not API_VAR_RE.match(str(self.name)):
            raise ValueError(f"api slot name must look like a1/a2..., got {self.name!r}.")
        if self.kind not in API_SLOT_KINDS:
            raise ValueError(f"api slot kind must be one of {API_SLOT_KINDS}, got {self.kind!r}.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "unit", str(self.unit))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, "target": self.target, "unit": self.unit}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ApiSlot":
        return cls(
            name=str(payload.get("name", "a1")),
            kind=str(payload.get("kind", "duration")),
            target=str(payload.get("target", "")),
            unit=str(payload.get("unit", "ns")),
        )


@dataclass(frozen=True)
class PulsePeriod:
    """One period-card in the pulse GUI."""

    duration: float | str
    states: tuple[int, ...]
    unit: str = "ns"
    name: str = ""

    def _duration_ns_unquantized(self, *, slots: Mapping[str, float] | None = None) -> tuple[float, str, float]:
        """Shared front half of :meth:`duration_steps` / :meth:`duration_ns`: evaluate the
        duration expression, validate the unit, and return ``(value, unit, ns)`` where ``value``
        is the raw quantity in ``unit`` and ``ns`` is it scaled to nanoseconds.

        ONLY the eval + unit-check is shared (single source for the ``unsupported pulse
        duration unit`` literal); each caller keeps its OWN boundary policy (negative/zero
        rejection, quantization) -- those are deliberate per-method differences, not duplication.
        ``value``/``unit`` are returned alongside ``ns`` so a caller can phrase an error in the
        source unit.
        """
        value = eval_time_expr(self.duration, slots=slots)
        unit = str(self.unit or "ns")
        if unit not in UNITS_TO_NS:
            raise ValueError(f"unsupported pulse duration unit {unit!r}.")
        return value, unit, value * UNITS_TO_NS[unit]

    def duration_steps(self, *, slots: Mapping[str, float] | None = None, time_step_ns: float = 1.0) -> int:
        value, unit, out = self._duration_ns_unquantized(slots=slots)
        # A NEGATIVE literal period duration is almost always an input error; raise instead of
        # silently snapping it up to one tick.  (Scan-table durations are clamped UI-side by
        # snap_scan_table; this guards the literal/period path, which validate() exercises.)
        if value < 0:
            raise ValueError(f"period duration must be >= 0 (got {value:g} {unit}).")
        return quantized_time_steps(out, time_step_ns=time_step_ns, allow_zero=False)

    def duration_ns(self, *, slots: Mapping[str, float] | None = None, time_step_ns: float | None = None) -> float:
        _value, _unit, out = self._duration_ns_unquantized(slots=slots)
        if time_step_ns is not None:
            return quantized_time_ns(out, time_step_ns=time_step_ns, allow_zero=False)
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
        api_slots: Sequence[ApiSlot | Mapping[str, object]] | None = None,
        time_step_ns: float = 1.0,
        repeat_start: int | None = None,
        repeat_end: int | None = None,
        repeat_count: int = 1,
        repeat_forever: bool = True,
        scan_repeats: int = 0,
        visible_channels: Sequence[str] | None = None,
        channel_labels: Mapping[str, str] | None = None,
        analog_buses: Mapping[str, Sequence[str]] | None = None,
        analog_bus_modes: Mapping[str, Sequence[Mapping[str, object] | str | None]] | None = None,
        clk_channels: Sequence[str] | None = None,
    ):
        self.channels = list(channel_names(channels, "channels"))
        # O(1) channel->bit index (channels are fixed after construction).  channel_index is
        # called per period x channel x rebuild from the GUI, so list.index() there is wasteful.
        self._channel_index = {channel: index for index, channel in enumerate(self.channels)}
        self.name = str(name) if name is not None else default_pulse_name()
        self.time_step_ns = positive_time_step_ns(time_step_ns)
        self.scan_slots = [slot if isinstance(slot, ScanSlot) else ScanSlot.from_dict(slot) for slot in (scan_slots or [])]
        # STRICT width check: a programmatic build must supply one column per bound slot.
        # Legacy short rows are tolerated only at the from_dict deserialization seam.
        self.scan_table = _normalize_scan_table(scan_table, slots=self.scan_slots)
        # API slots: named handles (a1/a2...) the API/Task set by name (set_api / state.aN).
        # Set EARLY so the ``state.aN = value`` attribute sugar (__setattr__) can find them.
        self.api_slots = [slot if isinstance(slot, ApiSlot) else ApiSlot.from_dict(slot) for slot in (api_slots or [])]
        self.periods = list(periods or default_periods(self.channels))
        self.delays = {str(k): v for k, v in dict(delays or {}).items()}
        self.delay_units = {str(k): str(v) for k, v in dict(delay_units or {}).items()}
        self.repeat_start = None if repeat_start is None else int(repeat_start)
        self.repeat_end = None if repeat_end is None else int(repeat_end)
        self.repeat_count = int(repeat_count)
        self.repeat_forever = bool(repeat_forever)
        # Number of FULL scan sweeps before the scan stops: 0 = sweep forever (the default,
        # matching the seamless cyclic streaming), K>=1 = play every scan point K times then
        # halt.  ORTHOGONAL to the measurement-layer camera ``repeat`` (frames per point); this
        # counts whole-table sweeps of the scan_table / api sweep.
        self.scan_repeats = max(0, int(scan_repeats))
        self.visible_channels = list(channel_names(visible_channels, "visible_channels", allow_empty=True)) if visible_channels is not None else default_visible_channels(self.channels)
        self.channel_labels = {str(k): str(v) for k, v in dict(channel_labels or {}).items()}
        self.analog_buses = {
            str(name): list(channel_names(members, f"analog bus {name!r}"))
            for name, members in dict(analog_buses or {}).items()
        }
        self.analog_bus_modes = self._normalize_analog_bus_modes(analog_bus_modes)
        # Channels wired directly to the FPGA clk (output = clk).  They are EXCLUDED from
        # the pulse engine (their edge-table bit is forced 0) so the engine never fights
        # the clk routing; the top muxes clk onto their pin via a runtime clk-enable mask.
        # Keep clk_channels RAW (do not silently drop unknown names) so validate() can flag a
        # typo / stale config instead of leaving clk quietly disabled.  Callers that intend to
        # drop clk channels missing from a new channel list (aligned_to_channels) pre-filter.
        self.clk_channels = [str(x) for x in (clk_channels or [])]
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
            if kind == "dac":
                # SIGNED value (0 = 0 V), like every user-facing DAC value.  The bound
                # field is the period's OWN target: an edge steps to it at the period
                # start, a RAMP reaches it at the period END -- so read the plan entry's
                # value, not the period-start (carried-in) level, which for a ramp is the
                # PREVIOUS period's value.  Only a hold (no value of its own) falls back
                # to the carried-in level.
                bus, period_index = target.split("@", 1)
                plan = self.analog_bus_plan(bus)
                entry = plan[int(period_index)] if int(period_index) < len(plan) else {}
                value = entry.get("value")
                if str(entry.get("mode", "hold")).strip().lower() in ("edge", "ramp") and value is not None:
                    slots = dict(slots)
                    if is_slot_ref(value):   # already bound elsewhere: its reference value
                        return float(slots.get(str(value).strip(), 0.0))
                    return float(value)
                return float(self.analog_bus_value_at_period_start(int(period_index), bus))
        except (ValueError, KeyError, IndexError, TypeError):
            pass   # malformed/out-of-range field -> fall back to the kind default (don't hide a real bug)
        return (1000.0 / scale) if kind == "duration" else 0.0

    def bind_field(self, kind: str, target: str, *, label: str = "", unit: str = "ns", nominal: float | None = None) -> int:
        """Bind a field to a new scan slot and rewrite the field to ``s{i}``.

        Returns the new slot index.  Idempotent: re-binding an already bound
        field returns its existing slot index.
        """

        if kind == "delay":
            raise ValueError("delay is a fixed per-channel value and cannot be scanned")
        existing = self.slot_index_for(kind, target)
        if existing is not None:
            return existing
        if nominal is None:
            nominal = self._read_field_nominal(kind, str(target), unit)
        # Duration slots are ALWAYS stored in ns: binding rewrites the field to
        # its "str (ns)" (ns) display, so a slot left in us/ms would scan in that unit while
        # the card shows "str (ns)" -- a silent 1000x mismatch.  Convert the nominal from the
        # field's entry unit to ns and pin the slot unit to ns so the Scan tab, the period
        # card, and the compiled program all agree.  (DAC slots keep their raw "value"
        # unit -- a DAC code is not a time.)
        if kind == "duration" and unit not in ("ns", "value"):
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

    def _set_bus_target(self, bus: str, period_index: int, value) -> None:
        """Write ``value`` into the DAC ``bus`` at ``period_index``, PRESERVING a ramp waveform but
        forcing any other mode to an ``edge`` -- the ONE place that 'keep ramp, else edge' rule + the
        ``analog_bus_modes`` plan writeback lives (scan-slot binding, api-field set, slot resolution),
        so the three callers can never drift apart (#B2).  ``value`` rides the plan entry verbatim: a
        str slot-var (a scan binding) or an int DAC code (resolved / api set) both work unchanged.
        The out-of-range fallback is unreachable (``validate`` rejects out-of-range periods)."""
        plan = self.analog_bus_plan(bus)
        cur = plan[period_index].get("mode", "hold") if period_index < len(plan) else "hold"
        mode = str(cur).strip().lower()
        plan[period_index] = {"mode": mode if mode == "ramp" else "edge", "value": value}
        self.analog_bus_modes[bus] = plan

    def _apply_slot_binding(self, index: int, slot: ScanSlot) -> None:
        var = slot_var(index)
        if slot.kind == "duration":
            period_index = int(slot.target)
            period = self.periods[period_index]
            self.periods[period_index] = PulsePeriod(var, period.states, unit="str (ns)", name=period.name)
        elif slot.kind == "dac":
            # PRESERVE the period's waveform mode (via _set_bus_target): a scanned EDGE steps to the
            # scanned value, a scanned RAMP ramps (from the carried-in value) TO the scanned value --
            # both hardware-seamless (the segment's stop endpoint reads the scan slot at runtime via
            # stop_value_select).  Only a HOLD becomes an edge when bound.  ``var`` is the str slot-var.
            self._set_bus_target(slot.dac_bus, slot.dac_period, var)

    def _clear_slot_binding(self, index: int, restore: float | str | None) -> None:
        slot = self.scan_slots[index]
        var = slot_var(index)
        if slot.kind == "duration":
            period_index = int(slot.target)
            period = self.periods[period_index]
            if str(period.duration) == var:
                value = 1_000 if restore is None else restore
                self.periods[period_index] = PulsePeriod(value, period.states, unit="ns", name=period.name)
        elif slot.kind == "dac":
            bus, period_index = slot.dac_bus, slot.dac_period
            plan = self.analog_bus_plan(bus)
            if period_index < len(plan) and str(plan[period_index].get("value")) == var:
                # Unbinding keeps the period's waveform mode (edge stays edge, ramp stays
                # ramp) and restores a concrete value: the caller's ``restore`` (the GUI
                # remembers the pre-binding text) or, failing that, the slot's nominal.
                mode = str(plan[period_index].get("mode", "hold")).strip().lower()
                value = restore if restore is not None else slot.nominal
                if mode in ("edge", "ramp") and value is not None:
                    plan[period_index] = {"mode": mode, "value": value}
                else:
                    plan[period_index] = {"mode": "hold", "value": None}
                self.analog_bus_modes[bus] = plan

    def _renumber_slot_bindings(self) -> None:
        # Map any field referencing an old s{k} to its new index after a removal.
        for new_index, slot in enumerate(self.scan_slots):
            self._apply_slot_binding(new_index, slot)

    # -- API slot helpers (named handles the API/Task set by name) ----------

    def api_names(self) -> list[str]:
        """API-slot handle names in slot order (``["a1", "a2", ...]``).  Names are unique
        per :class:`PulseTableState` (a duplicate would have been rejected by ``validate``),
        so the list is also the distinct set."""

        return [str(slot.name) for slot in self.api_slots]

    def api_slot_for(self, kind: str, target: str) -> str | None:
        """The API handle bound to field ``(kind, target)``, or ``None``."""

        for slot in self.api_slots:
            if slot.kind == kind and slot.target == str(target):
                return slot.name
        return None

    def _next_api_name(self) -> str:
        used = {int(m.group("index")) for s in self.api_slots if (m := API_VAR_RE.match(s.name))}
        index = 1
        while index in used:
            index += 1
        return api_var(index)

    def bind_api_field(self, kind: str, target: str, *, unit: str | None = None, name: str | None = None) -> str:
        """Tag field ``(kind, target)`` as an API slot and return its handle name.

        Idempotent: re-binding an already API-bound field returns its existing name.
        Does NOT change the field's value (the number stays); the slot is only a label
        the API sets by name.  A SCAN-bound field must be unbound first (a field is
        scanned OR api-settable, not both) -- the GUI cycle handles that ordering.
        """

        if kind not in API_SLOT_KINDS:
            raise ValueError(f"api slot kind must be one of {API_SLOT_KINDS}, got {kind!r}.")
        existing = self.api_slot_for(kind, str(target))
        if existing is not None:
            return existing
        if kind in SCAN_SLOT_KINDS and self.slot_index_for(kind, str(target)) is not None:
            raise ValueError(f"{kind} {target!r} is scan-bound; unbind its scan slot before making it an API slot.")
        name = str(name) if name else self._next_api_name()
        if not API_VAR_RE.match(name):
            raise ValueError(f"api slot name must look like a1/a2..., got {name!r}.")
        if unit is None:
            if kind == "dac":
                unit = "value"
            elif kind == "delay":
                # A bus target carries no entry in delay_units (that maps member
                # CHANNELS); read the first member's unit so a bus-delay slot gets a
                # sane unit instead of a bare "ns" that would mis-scale ms/s delays.
                source = self._delay_targets(str(target))[0]
                unit = self.delay_units.get(source, "ns")
            else:
                unit = "ns"
        self.api_slots.append(ApiSlot(name=name, kind=kind, target=str(target), unit=str(unit)))
        self.validate()
        return name

    def unbind_api_field(self, kind: str, target: str) -> "PulseTableState":
        """Remove the API slot on field ``(kind, target)`` (the value is untouched)."""

        self.api_slots = [s for s in self.api_slots if not (s.kind == kind and s.target == str(target))]
        self.validate()
        return self

    def set_api(self, name: str, value: float, *, unit: str | None = None) -> "PulseTableState":
        """Set the field tagged with API handle ``name`` to ``value``.

        The value is interpreted in the slot's own ``unit`` unless ``unit`` overrides it.
        Each name binds EXACTLY ONE field (names are unique within a state -- a duplicate
        is rejected by ``validate()``), so the data layer agrees with the GUI's per-cell
        dot allocator.  Raises if no field carries ``name`` (fail loud -- a silent no-op
        would hide a stale template)."""

        slot = next((s for s in self.api_slots if s.name == str(name)), None)
        if slot is None:
            raise ValueError(
                f"no API slot named {name!r}; this pulse exposes {self.api_names() or '[]'}. "
                "Tag a field as an API slot first (GUI: click a cell to a-state; API: bind_api_field).")
        self._set_api_field(slot, value, unit if unit is not None else slot.unit)
        return self

    def _set_api_field(self, slot: ApiSlot, value: float, unit: str) -> None:
        if slot.kind == "duration":
            self.set_period_duration(int(slot.target), float(value), unit=unit)
        elif slot.kind == "delay":
            # A delay target is either a single channel OR a DAC bus.  A bus's
            # delay field IS the shared per-member delay (the bus's bits ride one
            # delay line), so an API slot on it FANS OUT to every member channel --
            # the handle (a notebook's ``state.a1 = ...``) sets them uniformly,
            # exactly the symmetry a per-channel delay slot already has.
            targets = self._delay_targets(slot.target)
            for channel in targets:
                self.delays[channel] = float(value)
                self.delay_units[channel] = str(unit)
            self.validate()
        elif slot.kind == "dac":
            bus, period_index = slot.target.split("@", 1)
            self._set_bus_target(bus, int(period_index), int(round(float(value))))   # keep ramp, else edge (#B2)
            # Recompute the period DAC bit states NOW: ``to_sequence`` (the preview AND the virtual
            # backend's fired sequence) reads period.states, not the mode plan -- without this a
            # software api set/sweep of a DAC slot changed nothing on the virtual backend while the
            # real edge-table compiler (which reads the plan) did move, breaking virtual == real.
            self.apply_analog_bus_modes_to_period_states()
            self.validate()

    def _delay_targets(self, target: str) -> list[str]:
        """The channel(s) a delay slot's ``target`` writes/reads.  A plain channel is
        itself; a DAC-bus name fans out to its member channels (a bus's bits share one
        delay line).  Falls back to the literal target so a stale name never silently
        no-ops."""

        members = self.bus_channels(min_width=1).get(str(target))
        if members:
            return [str(channel) for channel in members]
        return [str(target)]

    def _read_api_field(self, slot: ApiSlot) -> float:
        if slot.kind == "duration":
            return float(self.periods[int(slot.target)].duration_ns(slots=self.reference_slots())) / UNITS_TO_NS.get(slot.unit, 1.0)
        if slot.kind == "delay":
            # For a bus the members are uniform after any set (an API slot IMPLIES
            # uniformity), so the first member's delay is the shared value.
            source = self._delay_targets(slot.target)[0]
            return eval_time_expr(self.delays.get(source, 0.0), slots=self.reference_slots())
        bus, period_index = slot.target.split("@", 1)
        plan = self.analog_bus_plan(bus)
        entry = plan[int(period_index)] if int(period_index) < len(plan) else {}
        return float(entry.get("value") or 0.0)

    def __setattr__(self, name: str, value: object) -> None:
        # Sugar: ``state.a1 = 1e-3`` sets the API slot named "a1" (only once the slot
        # exists; never shadows a real attribute -- no real field is named a<int>).
        if (API_VAR_RE.match(name) and "api_slots" in self.__dict__
                and any(s.name == name for s in self.api_slots)):
            self.set_api(name, value)  # type: ignore[arg-type]
            return
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> float:
        # Only invoked when normal lookup fails -> read an API slot by name.
        if API_VAR_RE.match(name):
            for slot in self.__dict__.get("api_slots", []):
                if slot.name == name:
                    return self._read_api_field(slot)
        raise AttributeError(name)

    def set_scan_table(self, rows: Sequence[Sequence[float]]) -> "PulseTableState":
        self.scan_table = _normalize_scan_table(rows, slots=self.scan_slots)
        self.validate()
        return self

    def load_scan_table(self, path: str | Path) -> "PulseTableState":
        return self.set_scan_table(load_scan_table(path, n_slots=len(self.scan_slots) or None))

    def clk_enable_mask(self) -> int:
        """Bitmask (bit n = channel ``self.channels[n]``) of channels wired to the FPGA
        clk.  The compiler forces these bits to 0 in the edge table and ships this mask so
        the top muxes the DAC strobe onto their pins (out_final[n] = clk_en[n] ? ~clk :
        engine_out[n]) -- the INVERTED clk so the DAC latches the parallel word at its
        data-eye centre (see zlc_pulse_streamer_top.v "DAC LATCH PHASE")."""

        mask = 0
        for channel in self.clk_channels:
            if channel in self.channels:
                mask |= 1 << self.channel_index(channel)
        return mask

    def is_clk_channel(self, channel: str) -> bool:
        return str(channel) in set(self.clk_channels)

    def scan_slot_dac_ranges(self) -> list[tuple[int, int] | None]:
        """Per-slot SIGNED DAC value range ``(-2^(B-1), +2^(B-1)-1)`` for ``dac``
        slots, ``None`` otherwise.  Aligned with :attr:`scan_slots`; pass to
        :func:`snap_scan_table` so DAC scan points are clamped to each bus's real
        signed range (0 = true 0 V on the offset-binary driver)."""

        buses = self.bus_channels(min_width=1)
        out: list[tuple[int, int] | None] = []
        for slot in self.scan_slots:
            if slot.kind == "dac":
                width = max(1, len(buses.get(slot.dac_bus, [])))
                out.append(bus_signed_range(width))
            else:
                out.append(None)
        return out

    def scan_column_specs(self) -> list["ScanColumnSpec"]:
        """Per scan-slot starter-sweep spec for the template (#scan-template): a DAC slot over its
        bus's SIGNED code range, a duration slot over a ns range bracketing the nominal (>= 1 tick).
        ONE source so the pulse GUI and the task-console pulse-scan form seed identical, per-kind
        columns -- a DAC column is never given a duration's ns sweep."""
        ranges = self.scan_slot_dac_ranges()
        step = positive_time_step_ns(self.time_step_ns)
        return [scan_column_spec(slot_var(index), slot.kind, nominal=slot.nominal, unit=slot.unit,
                                 signed_range=ranges[index], time_step_ns=step)
                for index, slot in enumerate(self.scan_slots)]

    def api_column_specs(self) -> list["ScanColumnSpec"]:
        """Per api-slot starter-sweep spec for the api-sweep template -- a DAC api slot over its bus's
        signed code range, a duration/delay api slot over a ns range (the SAME per-kind rule as the
        hardware scan, so neither template ever mixes a DAC's range with a duration's)."""
        buses = self.bus_channels(min_width=1)
        step = positive_time_step_ns(self.time_step_ns)
        specs: list[ScanColumnSpec] = []
        for slot in self.api_slots:
            rng = None
            if slot.kind == "dac":
                rng = bus_signed_range(max(1, len(buses.get(slot.target.split("@", 1)[0], []))))
            specs.append(scan_column_spec(slot.name, ("dac" if slot.kind == "dac" else "duration"),
                                          unit=slot.unit, signed_range=rng, time_step_ns=step))
        return specs

    def with_slots_resolved(self, slots: Mapping[str, float]) -> "PulseTableState":
        """Return a non-scan copy with each slot replaced by a constant value.

        Time slots take ns values; ``dac`` slots take SIGNED integer values
        (0 = true 0 V; the offset-binary wire code is produced by the compiler).
        Used for terse single-point notebook scans where one value is set per shot.
        """

        new = PulseTableState.from_dict(self.to_dict())
        # Missing slots default to their NOMINAL (reference) value, NOT 0 -- so a single-shot
        # resolve like set_time({"s0": ...}) only changes s0 and leaves the other slots at their
        # nominal, instead of silently zeroing those periods/DAC levels.
        resolved = new.reference_slots()
        resolved.update({str(k): float(v) for k, v in dict(slots or {}).items()})
        for index, slot in enumerate(new.scan_slots):
            value = float(resolved.get(slot_var(index), float(slot.nominal)))
            if slot.kind == "duration":
                period_index = int(slot.target)
                period = new.periods[period_index]
                new.periods[period_index] = PulsePeriod(value, period.states, unit="ns", name=period.name)
            elif slot.kind == "dac":
                # PRESERVE the period's waveform mode (same rule as _apply_slot_binding, via the ONE
                # _set_bus_target): resolving a RAMP-bound slot keeps it a ramp to the resolved value --
                # forcing "edge" made a single-point notebook run of a ramp-bound pulse play a step.
                new._set_bus_target(slot.dac_bus, slot.dac_period, int(round(value)))
        new.scan_slots = []
        new.scan_table = []
        new.apply_analog_bus_modes_to_period_states()
        new.validate()
        return new

    def with_api_resolved(self, values: Mapping[str, float]) -> "PulseTableState":
        """Return a copy with the named API slots set to ``values`` (each via :meth:`set_api`).

        This is the SOFTWARE analogue of :meth:`with_slots_resolved`: the hardware scan table
        streams scan slots on the FPGA, but API slots are fixed handles set per shot in software.
        A pulse-scan that sweeps an API slot deep-copies the base state and calls this per point,
        so each point fires a freshly-loaded pulse (load -> on_pulse -> wait -> next).  Unknown
        names raise (``set_api`` fails loud), exactly like the fixed-value path."""

        new = PulseTableState.from_dict(self.to_dict())
        for name, value in dict(values or {}).items():
            new.set_api(str(name), float(value))
        return new

    def primary_time_slot(self) -> str | None:
        """Return the variable name of the first duration (time) scan slot."""

        for index, slot in enumerate(self.scan_slots):
            if slot.is_time:
                return slot_var(index)
        return None

    def _resolve_step_ns(self, time_step_ns: float | None, clock_hz: float | None) -> float:
        """Resolve the tick grid (ns) from EITHER ``time_step_ns`` OR ``clock_hz`` (its
        reciprocal).  The rest of the API speaks ``clock_hz`` (``compile`` / ``compile_scan``
        / ``from_sequence`` / ``PulseSequence.validate`` all take it), so ``validate`` and
        ``to_sequence`` accept it too and convert ONCE here -- a caller never has to remember
        which of the two units a given method wants.  Defaults to the table's ``time_step_ns``."""
        if clock_hz is not None:
            if time_step_ns is not None:
                raise ValueError("pass either clock_hz or time_step_ns, not both.")
            time_step_ns = 1_000_000_000.0 / positive_float(clock_hz, "clock_hz")
        return self.time_step_ns if time_step_ns is None else positive_time_step_ns(time_step_ns)

    def validate(self, *, slots: Mapping[str, float] | None = None, time_step_ns: float | None = None,
                 clock_hz: float | None = None, validate_scan_slots: bool = True) -> "PulseTableState":
        # ``validate_scan_slots`` checks the slot bindings + the FULL N-row scan table; it is
        # SLOT-INDEPENDENT, so a per-scan-point validate (compile_scan) sets it False after
        # one full check -- otherwise validating N points each rescans the whole table, an
        # O(N^2) blow-up that dominated compile at thousands of points.
        # ``clock_hz`` is an ergonomic alias for ``time_step_ns`` (this table's tick grid).
        step_ns = self._resolve_step_ns(time_step_ns, clock_hz)
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
            # A DAC bus is multi-bit; a 1-channel "bus" is just a TTL channel and is only
            # half-supported (bus_value()/preview use min_width=2 and would not see it),
            # so reject it up front instead of letting it crash deeper.
            if len(members) < 2:
                raise ValueError(f"analog bus {name!r} must contain at least two channels.")
            unknown_members = [channel for channel in members if channel not in self.channels]
            if unknown_members:
                raise ValueError(f"analog bus {name!r} contains channels not in hardware channels: {unknown_members}.")
            bus_members.extend(members)
        duplicated_bus_members = sorted({channel for channel in bus_members if bus_members.count(channel) > 1})
        if duplicated_bus_members:
            raise ValueError(f"analog bus channels must not overlap: {duplicated_bus_members}.")
        known_buses = self.bus_channels(min_width=1)
        # An unknown clk channel is almost always a typo / stale config -- raise rather than
        # silently leave clk disabled on a pin the user thinks is clocked.
        unknown_clk = [c for c in self.clk_channels if c not in self.channels]
        if unknown_clk:
            raise ValueError(f"clk channels are not in hardware channels: {unknown_clk}.")
        # A clk channel is muxed directly onto the FPGA clk pin; if it were also an analog-bus
        # member the bus engine would drive that same DAC bit -> ambiguous double-drive on
        # hardware.  The GUI guards this, but validate() is the real contract gate (JSON /
        # notebook / from_dict bypass the GUI), so reject it here too.  Check against ALL
        # bus members -- inferred-from-labels AND explicit -- not just explicit analog_buses.
        all_bus_members = {channel for members in known_buses.values() for channel in members}
        clk_bus = sorted(set(self.clk_channels) & all_bus_members)
        if clk_bus:
            raise ValueError(f"clk channels must not be analog-bus members: {clk_bus}.")
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
            lo, hi = bus_signed_range(len(members))
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
                if is_slot_ref(value):
                    continue
                value_int = int(value)
                if value_int < lo or value_int > hi:
                    raise ValueError(
                        f"analog bus {bus_name!r} period {index} value must be between {lo} and {hi} "
                        "(signed LSB; 0 = 0 V on the offset-binary DAC driver).")
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
        self._validate_api_slots()
        return self

    def is_delay_target(self, target: str) -> bool:
        """True if ``target`` names a valid DELAY field: a single channel OR a DAC bus (a bus owns
        ONE delay that fans out to its members).  The ONE source both the validator AND the GUI's
        api-slot carry use -- so a bus-delay api slot is never wrongly dropped on a state rebuild
        (dropping it made the dot impossible to toggle OFF and let only one bus ever hold a slot)."""
        t = str(target)
        return t in self.channels or t in self.bus_channels(min_width=1)

    def _validate_api_slots(self) -> None:
        # Names are UNIQUE within a state (one handle = one field).  The GUI's per-cell dot
        # allocates a fresh ``a<N>`` each time, so duplicates only ever arise from a hand-edited
        # JSON or a misuse of bind_api_field -- which is precisely what this check catches.
        seen: set[str] = set()
        for slot in self.api_slots:
            if slot.name in seen:
                raise ValueError(
                    f"api slot name {slot.name!r} appears more than once; each handle "
                    "must bind exactly one field (GUI: every duration/delay/DAC cell gets its "
                    "own a<N>).")
            seen.add(slot.name)
        known_buses = self.bus_channels(min_width=1)
        for slot in self.api_slots:
            if slot.kind == "duration":
                period_index = int(slot.target) if slot.target.lstrip("-").isdigit() else -1
                if period_index < 0 or period_index >= len(self.periods):
                    raise ValueError(f"api slot {slot.name!r} binds duration of missing period {slot.target!r}.")
            elif slot.kind == "delay":
                # A delay target is a single channel OR a DAC bus (a bus owns a delay that
                # fans out to its members) -- both are valid, exactly the symmetry the user
                # asked for; anything else is a typo / stale config and fails loud.
                if not self.is_delay_target(slot.target):
                    raise ValueError(f"api slot {slot.name!r} binds delay of unknown channel/bus {slot.target!r}.")
            elif slot.kind == "dac":
                bus, _, period = slot.target.partition("@")
                if bus not in known_buses:
                    raise ValueError(f"api slot {slot.name!r} binds unknown analog bus {bus!r}.")
                if not period.lstrip("-").isdigit() or not (0 <= int(period) < len(self.periods)):
                    raise ValueError(f"api slot {slot.name!r} binds dac of missing period {period!r}.")

    def _validate_scan_slots(self) -> None:
        for index, slot in enumerate(self.scan_slots):
            if slot.kind == "duration":
                period_index = int(slot.target) if slot.target.lstrip("-").isdigit() else -1
                if period_index < 0 or period_index >= len(self.periods):
                    raise ValueError(f"scan slot {index} binds duration of missing period {slot.target!r}.")
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
            # A channel wired to the FPGA clk must survive an align onto the device
            # channel list (else it silently reverts to engine-driven and the clk pin
            # stops clocking) -- filter to the surviving channels like delays/labels.
            clk_channels=[channel for channel in self.clk_channels if channel in channels],
        )

    def label_for(self, channel: str) -> str:
        channel = self.channels[self.channel_index(channel)]
        return self.channel_labels.get(channel) or channel

    def channel_index(self, channel: str) -> int:
        try:
            return self._channel_index[str(channel)]
        except KeyError as exc:
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
        """Set one logical bus value (SIGNED LSB, 0 = true 0 V), updating the TTL bits.

        ``value`` is the user-facing SIGNED value (-2^(B-1) .. +2^(B-1)-1); the
        underlying member bits store the offset-binary code = value + bus_zero_code."""

        period_index = int(period_index)
        groups = self.bus_channels()
        if bus_name not in groups:
            raise ValueError(f"unknown bus channel {bus_name!r}.")
        members = groups[bus_name]
        lo, hi = bus_signed_range(len(members))
        value = int(value)
        if value < lo or value > hi:
            raise ValueError(f"{bus_name} must be between {lo} and {hi} (signed LSB; 0 = 0 V).")
        code = value + bus_zero_code(len(members))
        period = self.periods[period_index]
        states = list(period.states)
        for bit, channel in enumerate(members):
            states[self.channel_index(channel)] = 1 if (code >> bit) & 1 else 0
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
        # No explicit plan.  An UNTOUCHED bus (all member bits 0 in every period) rests
        # at true 0 V -- an all-hold plan (the hardware idles at the mid-scale code).
        # Nonzero bits are interpreted as offset-binary CODES (the bit-level truth),
        # so the derived plan values are SIGNED = code - bus_zero_code.
        zero_code = bus_zero_code(len(groups[bus_name]))
        codes = [self.bus_value(index, bus_name) for index in range(len(self.periods))]
        if all(code == 0 for code in codes):
            return [{"mode": "hold", "value": None} for _ in self.periods]
        out: list[dict[str, object]] = []
        previous: int | None = None
        for index, code in enumerate(codes):
            if index == 0 or previous is None or code != previous:
                out.append({"mode": "edge", "value": int(code) - zero_code})
            else:
                out.append({"mode": "hold", "value": None})
            previous = code
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
                # default to the SIGNED value currently encoded by the member bits
                members = self.bus_channels(min_width=1)[bus_name]
                value = self.bus_value(period_index, bus_name) - bus_zero_code(len(members))
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
        starts = self.period_start_steps(slots=slots, time_step_ns=self.time_step_ns)
        groups = self.bus_channels(min_width=1)
        for bus_name, members in groups.items():
            zero_code = bus_zero_code(len(members))
            code_max = (1 << len(members)) - 1
            for_each_plan = self._resolved_bus_plan(bus_name, slots)
            # An UNTOUCHED bus (all-hold plan, i.e. resting at 0 V) keeps its member bits
            # at 0 -- the "unused" marker -- instead of projecting the mid code into the
            # TTL bits (which would put phantom bits in masks/preview for a bus that does
            # nothing; the hardware idles at the mid code by itself, BUS_SAFE_VALUE).
            if all(str(entry.get("mode", "hold")).lower() == "hold" for entry in for_each_plan):
                continue
            for period_index, period in enumerate(self.periods):
                value = _analog_bus_value_at_tick(for_each_plan, starts, starts[period_index])
                # plan values are SIGNED (0 = 0 V); member bits store the offset-binary CODE
                code = max(0, min(code_max, int(value) + zero_code))
                states = list(period.states)
                for bit, channel in enumerate(members):
                    states[self.channel_index(channel)] = 1 if (code >> bit) & 1 else 0
                self.periods[period_index] = PulsePeriod(period.duration, tuple(states), unit=period.unit, name=period.name)
        return self

    def _resolved_bus_plan(self, bus_name: str, slots: Mapping[str, float]) -> list[dict[str, object]]:
        """Bus plan with slot references resolved to integers for preview."""

        plan = self.analog_bus_plan(bus_name)
        resolved: list[dict[str, object]] = []
        for entry in plan:
            value = entry.get("value")
            if is_slot_ref(value):
                value = int(round(float(slots.get(str(value), 0.0))))
            resolved.append({"mode": entry.get("mode", "hold"), "value": value})
        return resolved

    def analog_bus_value_at_period_start(self, period_index: int, bus_name: str) -> int:
        slots = self.reference_slots()
        starts = self.period_start_steps(slots=slots, time_step_ns=self.time_step_ns)
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

    def set_period_duration(self, period_index: int, duration: float | str, *, unit: str | None = None) -> "PulseTableState":
        """Set period ``period_index``'s duration (ns by default, or a scan-expr str).

        The programmatic equivalent of the pulse GUI's duration edit -- so a
        measurement can ``PulseTableState.load(...)`` a saved program and retune e.g. a
        readout duration without the GUI.  ``unit`` overrides the period's unit
        (``ns``/``us``/``ms``/``s``); omit to keep it."""
        period_index = int(period_index)
        if period_index < 0 or period_index >= len(self.periods):
            raise ValueError("period_index is out of range.")
        # A period whose duration is SCAN-BOUND carries an ``sN`` expression that a scan
        # slot targets; overwriting it would orphan that slot (a dead scan column).
        # Fail loud -- unbind it first (``unbind_slot``), as the GUI's dot toggle does.
        if self.slot_index_for("duration", str(period_index)) is not None:
            raise ValueError(
                f"period {period_index} duration is scan-bound; unbind its scan slot "
                "(unbind_slot) before setting a fixed duration.")
        period = self.periods[period_index]
        new_unit = str(unit) if unit is not None else period.unit
        # A LITERAL duration is floored to >= 1 whole tick HERE so a 0 / sub-tick value can never
        # reach the compiler/driver as a degenerate zero-length period (the user's "unexpected
        # error").  This is the SINGLE structural guard for "you cannot set a duration to 0": the
        # api-sweep path (_api_table_arrays) sets durations point-by-point with NO table-level snap,
        # so the floor must live at the set site.  A negative literal is left for validate() to
        # reject loudly (duration_steps); a scan EXPRESSION is left for the per-point scan snap.
        duration = self._floor_literal_duration(duration, new_unit)
        self.periods[period_index] = PulsePeriod(
            duration, tuple(period.states), unit=new_unit, name=period.name)
        self.validate()
        return self

    def _floor_literal_duration(self, duration: float | str, unit: str) -> float | str:
        """A literal duration in ``[0, 1 tick)`` is bumped UP to exactly one tick (the hardware
        minimum); a negative literal or a slot expression (needs ``sN``) is returned unchanged."""
        try:
            ns = float(eval_time_expr(duration)) * UNITS_TO_NS[str(unit)]
        except Exception:
            return duration                                  # an expression needing slots -- leave it
        step = positive_time_step_ns(self.time_step_ns)
        if 0.0 <= ns < step:                                 # 0 or sub-tick -> exactly 1 tick
            return step / UNITS_TO_NS.get(str(unit), 1.0)
        return duration

    def set_period_name(self, period_index: int, name: str) -> "PulseTableState":
        """Rename period ``period_index`` (the GUI's period-name edit)."""
        period_index = int(period_index)
        if period_index < 0 or period_index >= len(self.periods):
            raise ValueError("period_index is out of range.")
        period = self.periods[period_index]
        self.periods[period_index] = PulsePeriod(
            period.duration, tuple(period.states), unit=period.unit, name=str(name))
        self.validate()
        return self

    def set_channel_delay(self, channel: str, delay: float | str, *, unit: str = "ns") -> "PulseTableState":
        """Set the output delay (ns by default, or a scan-expr str) of ``channel`` OR a DAC
        BUS -- the programmatic form of the GUI's per-row delay edit.  A bus fans out to its
        member channels (the bus's bits share one delay line), exactly like the bus-delay API
        handle; a plain channel writes only itself."""
        channel = str(channel)
        targets = self._delay_targets(channel)
        if channel not in self._channel_index and channel not in self.bus_channels(min_width=1):
            raise ValueError(f"unknown channel/bus {channel!r}.")
        for ch in targets:
            self.delays[ch] = delay
            self.delay_units[ch] = str(unit)
        self.validate()
        return self

    def add_period(self, duration: float | str, *, name: str = "", unit: str = "ns",
                   states: Mapping[str, int] | Sequence[int] | None = None) -> "PulseTableState":
        """APPEND a new period (the GUI's add-period button).  ``states`` is a
        ``{channel: 0/1}`` map or a full per-channel sequence (omitted channels off).

        Append-only on purpose: inserting / removing / reordering periods would shift
        the period indices that scan slots and the repeat bracket reference, so those
        structural edits stay in the GUI (which reconciles them); a measurement that
        builds a program programmatically appends its steps in order."""
        if isinstance(states, Mapping):
            vec = [0] * len(self.channels)
            for channel, value in states.items():
                vec[self.channel_index(channel)] = 1 if int(value) else 0
        elif states is None:
            vec = [0] * len(self.channels)
        else:
            vec = [1 if int(v) else 0 for v in states]
            if len(vec) != len(self.channels):
                raise ValueError(f"states must have one value per channel ({len(self.channels)}).")
        self.periods.append(PulsePeriod(duration, tuple(vec), unit=str(unit), name=str(name)))
        # Keep per-period analog-bus mode entries in step with the new period count
        # (the same step the GUI's add-period does): a new period defaults to a HOLD
        # mode, then the period DAC states are recomputed from the modes -- else
        # ``validate()`` rejects the entries/periods length mismatch on ANY DAC-bus
        # program (the realistic neutral-atom case).
        self._grow_analog_bus_modes()
        self.apply_analog_bus_modes_to_period_states()
        self.validate()
        return self

    def _grow_analog_bus_modes(self) -> None:
        """Pad/trim every analog-bus mode list to one entry per period (a new period
        defaults to a ``hold`` mode), so the mode entries stay in step with
        ``self.periods`` after an :meth:`add_period`."""
        target = len(self.periods)
        for bus_name in list(self.analog_bus_modes):
            entries = [dict(e) if isinstance(e, Mapping) else e
                       for e in self.analog_bus_modes.get(bus_name, [])]
            if len(entries) < target:
                entries.extend({"mode": "hold", "value": None} for _ in range(target - len(entries)))
            elif len(entries) > target:
                entries = entries[:target]
            self.analog_bus_modes[bus_name] = entries

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

    def _remapped_target(self, kind: str, target: str) -> str:
        """Remap a slot ``target``'s PERIOD index through the bracket unroll (see
        :meth:`unrolled_bracket` / :meth:`_expand_bracket_index`).

        ``duration`` slots target a bare period index; ``dac`` slots target
        ``"<bus>@<period_index>"`` -- both have their period index expanded.  A ``delay``
        slot targets a CHANNEL name (and a ``duration`` target that is not numeric, e.g. an
        expression) carries no period index, so it returns unchanged.  This is the single
        source for that one index-remap rule, shared by the scan-slot and api-slot loops;
        each loop keeps its own (different) serialization.
        """
        if kind == "duration" and str(target).lstrip("-").isdigit():
            return str(self._expand_bracket_index(int(target)))
        if kind == "dac":
            bus, period_index = target.split("@", 1)
            return f"{bus}@{self._expand_bracket_index(int(period_index))}"
        return target

    def unrolled_bracket(self) -> "PulseTableState":
        """Return a NEW state with the inner finite repeat bracket fully UNROLLED into a
        flat period list (``repeat_count`` becomes 1, the bracket is cleared).

        This is the unifying trick that makes a constant channel delay work with an
        inner bracket: once the bracket is flat there is no inner-loop boundary
        for an additively-shifted edge to cross, so the existing flat machinery (additive
        delay + affine scan + repeat_forever) handles delays in ANY form.  The whole
        flat frame can still repeat via ``repeat_forever``.

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
            payload["target"] = self._remapped_target(slot.kind, slot.target)
            scan_slots.append(payload)

        api_slots: list[dict[str, object]] = []
        for slot in self.api_slots:
            payload = slot.to_dict()
            payload["target"] = self._remapped_target(slot.kind, slot.target)   # delay -> channel name, unchanged
            api_slots.append(payload)

        return type(self)(
            channels=list(self.channels),
            periods=[PulsePeriod(p.duration, p.states, unit=p.unit, name=p.name) for p in expand(self.periods)],
            delays=dict(self.delays),
            delay_units=dict(self.delay_units),
            name=self.name,
            scan_slots=scan_slots,
            # api_slots carry through the unroll just like scan_slots (built above with the same
            # target remap): a bracketed period's api-bound duration is duplicated to every copy
            # (each carries the same api expression, so they resolve in lockstep), and the binding
            # points at the first copy.  Omitting this dropped the api-slot bindings on any
            # finite-bracket compile -- the same built-but-not-passed bug class as clk_channels below.
            api_slots=api_slots,
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
            # channels are unchanged by the unroll, so the clk set carries verbatim -- without
            # this a finite-bracket-with-delay compile (which unrolls first) would silently drop
            # clk channels back to engine-driven (same bug class as aligned_to_channels).
            clk_channels=list(self.clk_channels),
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

    def period_start_steps(self, *, slots: Mapping[str, float] | None = None, time_step_ns: float) -> list[int]:
        """Prefix sum of period durations: ``starts[i]`` is the tick at which period ``i`` begins
        (``starts[-1]`` == total frame duration).  ``time_step_ns`` is REQUIRED (never falls back to
        ``self.time_step_ns``) so the same source serves both the preview (``self.time_step_ns``) and
        the hardware compiler (``clock_step_ns = 1e9 / clock_hz``).  The affine compiler
        (``_pulse_table_affine_period_starts``) is the affine form of this same prefix sum."""
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
        clock_hz: float | None = None,
        expand_repeat: bool = True,
    ) -> PulseSequence:
        step_ns = self._resolve_step_ns(time_step_ns, clock_hz)   # clock_hz = ergonomic alias for time_step_ns
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
        slots: Mapping[str, float] | None = None,
        repeat_forever: bool | None = None,
    ):
        from ..devices.sequencer import compile_pulse_table_runtime_program

        clock_hz = positive_float(clock_hz, "clock_hz")
        return compile_pulse_table_runtime_program(
            self,
            channels=self.channels,
            clock_hz=clock_hz,
            slots=slots,
            repeat_forever=self.repeat_forever if repeat_forever is None else bool(repeat_forever),
        )

    def compile_scan(
        self,
        *,
        clock_hz: float,
        scan_table: Sequence[Sequence[float]] | None = None,
        repeat_forever: bool | None = None,
    ):
        from ..devices.sequencer import compile_pulse_table_scan_runtime_program

        clock_hz = positive_float(clock_hz, "clock_hz")
        step_ns = 1_000_000_000.0 / clock_hz
        # Snap + clamp before compiling so the program is exactly what the hardware can
        # run, no matter how compile_scan is reached (GUI, notebook, save bundle):
        # durations -> whole ticks, fixed delays -> ticks, DAC scan codes -> [0, max].
        source = self
        if scan_table is not None:
            source = PulseTableState.from_dict(self.to_dict())
            source.set_scan_table(scan_table)
        snapped = source.snapped(time_step_ns=step_ns)
        return compile_pulse_table_scan_runtime_program(
            snapped,
            channels=snapped.channels,
            clock_hz=clock_hz,
            scan_table=snapped.scan_table,
            repeat_forever=snapped.repeat_forever if repeat_forever is None else bool(repeat_forever),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "name": self.name,
            "channels": list(self.channels),
            "scan_slots": [slot.to_dict() for slot in self.scan_slots],
            "scan_table": [list(row) for row in self.scan_table],
            "api_slots": [slot.to_dict() for slot in self.api_slots],
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
            "scan_repeats": int(self.scan_repeats),
            "clk_channels": list(self.clk_channels),
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
        # A per-channel delay past +/- DELAY_MAX_TICKS can never be realized; do NOT
        # silently clamp it here (that would corrupt the physics) -- the GUI clamps the
        # input field, and the compiler raises a clear DelayTooLargeError at validate.
        copy.scan_table = snap_scan_table(
            copy.scan_table, copy.scan_slots, time_step_ns=step, dac_ranges=copy.scan_slot_dac_ranges()
        )
        copy.validate()
        return copy

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PulseTableState":
        if payload.get("schema", cls.schema) != cls.schema:
            raise ValueError("unsupported pulse table schema.")
        # The ONE legacy-tolerance seam for short scan-table rows.  from_dict is where every
        # PERSISTED payload re-enters the object model (load()ed .json pulse files, saved-figure
        # recipes, server pulse transfers), and only an old save written before another slot was
        # bound legitimately carries short rows.  Bare __init__ cannot tell a saved payload from
        # a programmatic build, so __init__/set_scan_table stay strict and the pad lives here:
        # missing columns take the bound slot's NOMINAL (bind_field's rule, never 0.0) and a
        # warning names the payload.  In-process to_dict() round-trips (snapped(), with_*_resolved)
        # always carry full-width rows, so the pad never fires for them.
        scan_slots = [
            slot if isinstance(slot, ScanSlot) else ScanSlot.from_dict(slot)
            for slot in payload.get("scan_slots", ())
        ]
        scan_table = _normalize_scan_table(
            payload.get("scan_table", ()),
            slots=scan_slots,
            pad_legacy_source=str(payload.get("name", "")) or "<unnamed>",
        )
        return cls(
            name=str(payload["name"]) if "name" in payload else None,
            channels=payload["channels"],
            scan_slots=scan_slots,
            scan_table=scan_table,
            api_slots=payload.get("api_slots", ()),
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
            scan_repeats=int(payload.get("scan_repeats", 0)),
            clk_channels=payload.get("clk_channels"),
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
    def from_sequence(cls, sequence: PulseSequence, *, channels: Sequence[str], clock_hz: float = DEFAULT_CLOCK_HZ) -> "PulseTableState":
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


def default_imaging_template(
    channels: Sequence[str] | None = None,
    *,
    cooling: float = 2e-3,
    reference_exposure: float = 20e-3,
    readout_exposure: float = 5e-3,
    gap: float = READOUT_GAP_SECONDS,
    trap_channel: str = "trap",
    cooling_channel: str = "cooling",
    probe_channel: str = "probe",
    trigger_channel: str = "emCCD",
) -> "PulseTableState":
    """The real, inspectable imaging program the Calibrate task loads -- a literal
    CONTINUOUS long-short-long bracket (NOT a single window the task secretly unrolls).

    One ``load`` (cooling) cycle, then three back-to-back camera exposures on the SAME
    atoms -- ``image_0`` (long reference), ``image_1`` (short readout), ``image_2`` (long
    reference) -- each its own emCCD trigger, separated by a ``gap`` that HOLDS ONLY the
    trap (cooling/probe/trigger off) so the camera falls and re-arms between frames WITHOUT
    re-cooling (re-cooling would scramble the atoms and void the reference labels).

    Three API handles -- ``a1`` (first long reference), ``a2`` (short readout), ``a3``
    (second long reference) -- one per exposure cell, just like the pulse GUI allocates a
    fresh ``a<N>`` per click.  The Calibrate task sets all three BY NAME
    (``set_api("a1", long); set_api("a2", short); set_api("a3", long)``); it only changes
    those durations, never the structure.  What you load IS what is fired: open this
    template in the pulse GUI and you see exactly the long-short-long the task runs."""

    chans = list(channels) if channels else [trap_channel, cooling_channel, probe_channel, trigger_channel]

    def states(active) -> tuple[int, ...]:
        return tuple(1 if ch in active else 0 for ch in chans)

    image = states({trap_channel, probe_channel, trigger_channel})
    held = states({trap_channel})   # gap: trap held, NO cooling -> no re-cool between frames
    periods = [
        PulsePeriod(float(cooling), states({trap_channel, cooling_channel}), unit="s", name="load"),
        PulsePeriod(float(reference_exposure), image, unit="s", name="image_0"),
        PulsePeriod(float(gap), held, unit="s", name="gap_0"),
        PulsePeriod(float(readout_exposure), image, unit="s", name="image_1"),
        PulsePeriod(float(gap), held, unit="s", name="gap_1"),
        PulsePeriod(float(reference_exposure), image, unit="s", name="image_2"),
    ]
    state = PulseTableState(channels=chans, periods=periods, name="imaging_template")
    # One API handle per exposure cell (names are unique, like the GUI allocates):
    # a1 = first long reference (image_0), a2 = short readout (image_1), a3 = second long
    # reference (image_2).  The Calibrate task sets all three by name.
    state.bind_api_field("duration", "1", name="a1", unit="s")
    state.bind_api_field("duration", "3", name="a2", unit="s")
    state.bind_api_field("duration", "5", name="a3", unit="s")
    return state


def single_imaging_template(
    channels: Sequence[str] | None = None,
    *,
    cooling: float = 2e-3,
    exposure: float = 5e-3,
    trap_channel: str = "trap",
    cooling_channel: str = "cooling",
    probe_channel: str = "probe",
    trigger_channel: str = "emCCD",
) -> "PulseTableState":
    """A SINGLE-shot imaging program (``load`` -> one ``image`` frame, ONE camera trigger) --
    the base pulse for a generic single-image measurement (e.g. the Pulse-scan, which images
    ONCE per scan point), as opposed to the Calibrate task's long-short-long
    :func:`default_imaging_template` (three triggers).  The ``image`` duration is API slot
    ``a1`` so a notebook/API can set the exposure by name."""

    chans = list(channels) if channels else [trap_channel, cooling_channel, probe_channel, trigger_channel]

    def states(active) -> tuple[int, ...]:
        return tuple(1 if ch in active else 0 for ch in chans)

    periods = [
        PulsePeriod(float(cooling), states({trap_channel, cooling_channel}), unit="s", name="load"),
        PulsePeriod(float(exposure), states({trap_channel, probe_channel, trigger_channel}), unit="s", name="image"),
    ]
    state = PulseTableState(channels=chans, periods=periods, name="probe_template")
    state.bind_api_field("duration", "1", name="a1", unit="s")   # image exposure, settable by name
    return state


def resolve_pulse_template(template, *, default_name: str, default_factory) -> "PulseTableState":
    """The SINGLE resolver every form that takes a pulse-template path goes through (the Calibrate
    task + the Pulse-scan measurement).  Load order: the given path if it is a real file; else the
    same-named file shipped under the project ``pulses/`` folder -- anchored to the PROJECT ROOT via
    ``project_path`` (NOT a fragile ``parents[N]`` count that breaks when a caller moves); else the
    in-memory ``default_factory()`` (e.g. the long-short-long or single-image default)."""
    from Zou_lab_control._paths import project_path        # absolute import: the dependency-free path seam
    text = str(template or "").strip() or default_name
    path = Path(text)
    if path.is_file():
        return PulseTableState.load(path)
    name = path.name
    for base in (Path("pulses"), project_path("pulses")):     # CWD-relative, then project-anchored
        shipped = base / name
        if shipped.is_file():
            return PulseTableState.load(shipped)
    return default_factory()


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


def bus_period_levels(
    plan: Sequence[Mapping[str, object]], starts: Sequence[int], *, looping: bool = False
) -> list[tuple[int, int, int, int, str]]:
    """Per-period DAC levels with the WITHIN-PERIOD waveform semantics.

    Walks the periods front-to-back carrying the current DAC value.  ``looping`` chooses the value
    entering period 0: a single fire enters at idle 0 V (the default); a LOOPING program enters at the
    steady-state the loop converges to -- the level of the last edge/ramp the frame leaves -- so a
    looping ``[ramp V, hold V]`` shows FLAT V (the hardware carries V across the wrap), not the one-time
    idle->V first frame (#ramp-carry).  For each period returns
    ``(start_tick, stop_tick, in_value, out_value, mode)`` where ``in_value`` is the
    value entering the period and:

    * ``edge v`` -> the period steps to ``v`` at its start and holds it
      (``out_value = v``);
    * ``ramp v`` -> the period ramps linearly from ``in_value`` at ``start_tick`` to
      ``v`` at ``stop_tick`` (``out_value = v``);
    * ``hold``   -> the period holds the carried-in value (``out_value = in_value``).

    This is the SINGLE source of truth for the DAC waveform, shared by the preview
    value/tick helpers and (in structure) the hardware segment compiler -- so a ramp
    always describes the CURRENT period, and a hold always shows the value carried in
    from whatever edge/ramp preceded it (which updates if that upstream value changes)."""

    count = min(len(plan), max(0, len(starts) - 1))
    levels: list[tuple[int, int, int, int, str]] = []
    carried = 0
    if looping:
        # seed period 0 with the steady-state the loop converges to = the level of the LAST edge/ramp
        # the frame leaves (edge/ramp set an absolute level; a trailing hold keeps it carried) (#ramp-carry).
        for entry in plan[:count]:
            mode = str(entry.get("mode", "hold")).strip().lower()
            value = entry.get("value")
            if value is not None and mode in {"edge", "ramp"}:
                carried = int(value)
    for index in range(count):
        entry = plan[index]
        mode = str(entry.get("mode", "hold")).strip().lower()
        value = entry.get("value")
        start_tick = int(starts[index])
        stop_tick = int(starts[index + 1])
        in_value = carried
        if value is not None and mode in {"edge", "ramp"}:
            out_value = int(value)
        else:
            mode = "hold"
            out_value = carried
        levels.append((start_tick, stop_tick, in_value, out_value, mode))
        carried = out_value
    return levels


def _analog_bus_value_at_tick(plan: Sequence[Mapping[str, object]], starts: Sequence[int], tick: int,
                              *, looping: bool = False) -> int:
    tick = int(tick)
    levels = bus_period_levels(plan, starts, looping=looping)
    if not levels:
        return 0
    for start_tick, stop_tick, in_value, out_value, mode in levels:
        if start_tick <= tick < stop_tick:
            if mode == "ramp" and stop_tick > start_tick:
                # The hardware ramp engine is a Bresenham stepper: after k ticks it has
                # moved floor(k*|delta|/span) codes from the carried-in value (multiple
                # LSBs per tick for steep ramps), landing exactly on the target at
                # stop_tick.  The preview draws that same integer staircase -- works
                # unchanged in this SIGNED user domain (the offset cancels in delta).
                span = int(stop_tick) - int(start_tick)
                delta = abs(int(out_value) - int(in_value))
                k = int(tick) - int(start_tick)
                moves = min(delta, (k * delta) // span)
                return int(in_value) + (moves if out_value >= in_value else -moves)
            return int(out_value)
    # at/after the table end the bus holds its final level
    if tick >= levels[-1][1]:
        return int(levels[-1][3])
    return 0


def analog_bus_ticks(plan: Sequence[Mapping[str, object]], starts: Sequence[int]) -> list[int]:
    """Breakpoint ticks for drawing/expanding the bus waveform: every period boundary
    plus sample ticks inside each ramp period (at most one per tick -- a steep Bresenham
    ramp moves multiple LSBs per tick, so per-tick sampling captures every level)."""

    levels = bus_period_levels(plan, starts)
    ticks = {0}
    for start_tick, stop_tick, in_value, out_value, mode in levels:
        ticks.add(start_tick)
        if mode == "ramp" and stop_tick > start_tick and in_value != out_value:
            span = stop_tick - start_tick
            steps = abs(out_value - in_value)
            last = start_tick
            for step in range(1, steps + 1):
                tick = int(round(start_tick + span * (step / steps)))
                tick = max(start_tick, min(stop_tick, tick))
                if tick <= last and last < stop_tick:
                    tick = last + 1
                if tick <= stop_tick:
                    ticks.add(tick)
                    last = tick
    if starts:
        ticks.add(int(starts[-1]))
    return sorted(ticks)


def is_slot_ref(value: object) -> bool:
    """True when ``value`` is a scan-slot reference like ``"s0"`` / ``"s3"``.

    The SINGLE slot-reference parser (shared by the sequencer compiler and the GUI so
    the ``sN`` spelling cannot drift between layers)."""
    return isinstance(value, str) and bool(SLOT_VAR_RE.fullmatch(value.strip()))


def slot_ref_index(value: object) -> int | None:
    """Return the slot index N for a ``"sN"`` reference, else ``None``."""
    if not isinstance(value, str):
        return None
    match = SLOT_VAR_RE.fullmatch(value.strip())
    return int(match.group("index")) if match else None


def _coerce_bus_value(value: object) -> object:
    if value is None:
        return None
    if is_slot_ref(value):
        return value.strip()
    return int(value)


def _normalize_scan_table(
    rows: Sequence[Sequence[float]] | None,
    *,
    slots: Sequence["ScanSlot"],
    pad_legacy_source: str | None = None,
) -> list[list[float]]:
    """Coerce ``rows`` to floats and REJECT any width mismatch against the bound ``slots``.

    Strict in BOTH directions by default: a too-wide row would silently drop a column, and
    a too-short row would silently scan a wrong value for the unfilled slot(s) -- either way
    the hardware runs a different experiment than the user asked for, so both are hard errors
    with the offending row named.

    ``pad_legacy_source`` is the ONE deliberate tolerance, reserved for deserializing
    PERSISTED payloads (:meth:`PulseTableState.from_dict`): a save written before another
    slot was bound legitimately carries short rows.  When set (to a human-readable payload
    name for the warning), short rows are padded with each missing slot's NOMINAL value --
    the same "a new scan dimension starts at the field's current value" rule
    :meth:`PulseTableState.bind_field` uses, never 0.0 -- and a warning is logged.  Too-wide
    rows still raise even then (there is no legacy writer of wide rows; that is data loss).
    """
    if rows is None:
        return []
    n_slots = len(slots)
    slot_names = ", ".join(
        f"{slot_var(i)}={slot.label or f'{slot.kind}:{slot.target}'}" for i, slot in enumerate(slots)
    )
    out: list[list[float]] = []
    padded_rows: list[int] = []
    for index, row in enumerate(rows):
        if isinstance(row, Number):
            values = [float(row)]
        else:
            values = [float(value) for value in row]
        if n_slots and len(values) != n_slots:
            if len(values) > n_slots:
                raise ValueError(
                    f"scan table row {index} has {len(values)} values but only {n_slots} scan "
                    f"slot(s) are bound ({slot_names}); give exactly one column per bound slot."
                )
            if pad_legacy_source is None:
                raise ValueError(
                    f"scan table row {index} has {len(values)} values but {n_slots} scan slot(s) "
                    f"are bound ({slot_names}); a short row would silently scan a wrong value for "
                    f"the missing slot(s) -- give exactly one column per bound slot."
                )
            values = values + [float(slot.nominal) for slot in slots[len(values):]]
            padded_rows.append(index)
        out.append(values)
    if padded_rows:
        logging.getLogger(__name__).warning(
            "pulse table %r: scan table row(s) %s are narrower than the %d bound scan slot(s) "
            "(%s); padded the missing column(s) with each slot's nominal value (legacy save).",
            pad_legacy_source, padded_rows, n_slots, slot_names,
        )
    return out


def load_scan_table(path: str | Path, *, n_slots: int | None = None) -> list[list[float]]:
    """Load a scan table (``N_points x N_slots``) from ``.npy``/``.csv``/``.txt``.

    ``.npy`` is read with NumPy.  Text files accept comma or whitespace
    separators and ignore ``#`` comment lines and a single header line of names.

    A 1-D array is ambiguous (``[1, 2, 3]`` could be 1 point of 3 slots or 3 points
    of 1 slot).  When ``n_slots`` is given it disambiguates: a flat array whose length
    is a multiple of ``n_slots`` is reshaped to ``(-1, n_slots)`` -- so ``n_slots=1``
    gives a COLUMN (3 points x 1 slot), the intuitive single-slot case, and
    ``n_slots=2`` over ``[a, b, c, d]`` gives 2 points.  Without ``n_slots`` a 1-D array
    stays a single row (legacy behavior)."""

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
    array = np.asarray(array, dtype=float)
    # Disambiguate a 1-D array by the known slot count: a flat array whose length is a
    # multiple of n_slots is N points x n_slots (so n_slots=1 -> a column of points).
    if array.ndim == 1 and n_slots and int(n_slots) > 0 and array.size % int(n_slots) == 0:
        array = array.reshape(-1, int(n_slots))
    array = np.atleast_2d(array)
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
        number * factor, time_step_ns=time_step_ns,
        allow_zero=allow_zero, allow_negative=allow_negative,
    )
    out = snapped_ns / factor
    return int(out) if float(out).is_integer() else out


#: Default DAC bus width (bits) used to clamp a DAC scan point when no per-slot
#: maximum is supplied.  Matches the 10-bit buses in the top/XDC.
DEFAULT_DAC_BITS = 10


def snap_scan_table(
    scan_table: Sequence[Sequence[float]],
    scan_slots: Sequence["ScanSlot"],
    *,
    time_step_ns: float,
    dac_ranges: Sequence[tuple[int, int] | None] | None = None,
) -> list[list[float]]:
    """Snap + clamp every scan-table point to a value the hardware can actually run.

    Per slot kind, with the SAME rules the rest of the toolchain enforces, so the
    table the user sees / saves / uploads is exactly what the FPGA plays:

    * ``duration`` -> the nearest whole clock tick, snapped UP to at least one tick
      (a period must occupy >= 1 tick) and never negative.
    * ``dac`` -> the nearest SIGNED integer (0 = true 0 V), clamped to
      ``dac_ranges[j]`` for that slot if given, else the default 10-bit signed range
      ``(-2^(B-1), +2^(B-1)-1)`` = (-512, +511).  The offset-binary wire code is
      produced later, by the compiler (code = signed + 2^(B-1)).

    ``dac_ranges`` (optional) is a per-slot sequence aligned with ``scan_slots``;
    entries for non-DAC slots are ignored.  One shared snap source for the GUI (so the
    displayed/saved table matches) and the server/pulse API (so the transferred pulse
    matches the hardware)."""

    step = positive_time_step_ns(time_step_ns)
    default_range = bus_signed_range(DEFAULT_DAC_BITS)
    # Check the column count against the slot count FIRST (a mismatch in EITHER direction
    # raises) -- otherwise a mismatched loaded array would be silently truncated/under-snapped
    # by the zip() below (a column dropped, or a slot left un-clamped).  This is the
    # on-hardware fire path, so "loading the wrong-width array" must be a clear error, never
    # a silently different experiment.
    normalized = _normalize_scan_table(scan_table, slots=list(scan_slots))
    out: list[list[float]] = []
    for row in normalized:
        new_row: list[float] = []
        for index, (value, slot) in enumerate(zip(row, scan_slots)):
            if slot.kind == "dac":
                rng = None
                if dac_ranges is not None and index < len(dac_ranges):
                    rng = dac_ranges[index]
                lo, hi = default_range if rng is None else (int(rng[0]), int(rng[1]))
                signed = int(round(float(value)))
                new_row.append(float(max(lo, min(hi, signed))))
            else:
                # A scanned period duration must be >= 1 tick and never negative.
                snapped = _snap_literal_time_value(
                    float(value), slot.unit, step, allow_zero=False, allow_negative=False
                )
                new_row.append(float(snapped))
        out.append(new_row)
    return out


@dataclass(frozen=True)
class ScanColumnSpec:
    """One column of a starter scan/api template: the column variable name + a KIND-APPROPRIATE
    default sweep range.  A ``dac`` column sweeps INTEGER codes over the bus's signed range
    (``0`` = 0 V); a time column sweeps its unit (ns by default), ``>= 1`` tick.  This is why a
    DAC slot no longer inherits a duration's ns range -- the bug the operator hit, where a +-512
    DAC column was seeded with a ``20 .. 200000`` ns sweep (every point then clamped to +511)."""

    name: str
    lo: float
    hi: float
    is_dac: bool = False
    unit: str = "ns"


def scan_column_spec(name: str, kind: str, *, nominal: float = 0.0, unit: str = "ns",
                     signed_range: tuple[int, int] | None = None, time_step_ns: float = 20.0) -> ScanColumnSpec:
    """Build a :class:`ScanColumnSpec` with a sensible per-kind default sweep.

    ``dac`` -> the bus's SIGNED code range (``signed_range``, else the default 10-bit ``+-512``).
    ``duration`` / time -> a ns range bracketing ``nominal`` (a few x), floored at one tick so the
    column is never seeded with a 0 / sub-tick duration (the driver rejects a zero-length period)."""
    if str(kind) == "dac":
        lo, hi = signed_range if signed_range is not None else bus_signed_range(DEFAULT_DAC_BITS)
        return ScanColumnSpec(name, float(lo), float(hi), is_dac=True, unit=str(unit or "code"))
    u = UNITS_TO_NS.get(str(unit or "ns"), 1.0)
    step_in_unit = positive_time_step_ns(time_step_ns) / u           # one clock tick, in the slot's unit
    nom = max(0.0, float(nominal))
    lo = step_in_unit                                                # 1 tick minimum -- never 0
    hi = max(nom * 2.0, 100.0 * step_in_unit)                        # bracket the nominal, >= ~100 ticks
    return ScanColumnSpec(name, float(lo), float(hi), is_dac=False, unit=str(unit or "ns"))


def scan_table_template(kind: str, columns: Sequence[ScanColumnSpec]) -> str:
    """Starter Python for a scan-table program -- the ONE scan model, shared by the pulse GUI
    Scan tab and the task-console Pulse-scan form.

    The program builds an ``(N_points x n_cols)`` array and assigns it to ``scan_table``: one ROW
    per scan point, one COLUMN per bound slot.  The whole table is one object, so the slots advance
    together (lockstep); correlations (anti-correlated, grid, loaded array) are just different ways
    of building that one array.  Each column is seeded by its slot's KIND (``columns`` carries the
    per-slot range): a DAC column sweeps integer codes over its signed range, a duration column
    sweeps ns ticks bracketing the nominal -- so a DAC slot is NOT given a duration's ns range.

    * ``column_stack`` (default): one independent column per slot.
    * ``grid``: every combination (outer product) of per-axis arrays.
    """

    cols = list(columns) or [ScanColumnSpec("s0", 20.0, 200_000.0, is_dac=False, unit="ns")]
    n = len(cols)

    def sweep(spec: ScanColumnSpec, size) -> str:
        base = f"np.linspace({spec.lo:g}, {spec.hi:g}, {size})"
        return f"{base}.round().astype(int)" if spec.is_dac else base

    def note(spec: ScanColumnSpec) -> str:
        return (f"{spec.name}: DAC code [{spec.lo:g}..{spec.hi:g}], 0 = 0 V"
                if spec.is_dac else f"{spec.name}: duration [{spec.unit}], >= 1 tick")

    if str(kind) == "grid":
        # A real N-D grid: ONE axis per slot, every combination (outer product).  Each axis is seeded
        # in its own unit/range; scan_shape lets the grid show as a scan map.  Modest default sizes.
        sizes = [5, 4, 3] + [2] * max(0, n - 3)
        lines = ["import numpy as np", "",
                 f"# Grid scan over {n} slot(s) {cols[0].name}..{cols[-1].name}: every combination of the per-slot axes."]
        for j, spec in enumerate(cols):
            lines.append(f"a{j} = {sweep(spec, sizes[j])}        # axis for {note(spec)}")
        mesh = ", ".join(f"A{j}" for j in range(n))
        axes = ", ".join(f"a{j}" for j in range(n))
        ravel = ", ".join(f"A{j}.ravel()" for j in range(n))
        shape = ", ".join(f"len(a{j})" for j in range(n))
        shape_expr = f"({shape},)" if n == 1 else f"({shape})"   # always a tuple, even for n == 1
        lines.append(f"{mesh}, = np.meshgrid({axes}, indexing=\"ij\")")
        lines.append(f"scan_table = np.column_stack([{ravel}])")
        lines.append(f"scan_shape = {shape_expr}        # {n}-D grid -> a scan map")
        return "\n".join(lines) + "\n"
    # column_stack: one independent column per slot, the columns advancing together (lockstep).
    lines = ["import numpy as np", "",
             f"# {n} bound slot(s) {cols[0].name}..{cols[-1].name}: build an (N_points x {n}) array -- one row",
             "# per scan point, one column per slot (each in its OWN unit: ns for a duration, integer",
             "# code for a DAC).  Edit each column; the columns advance together (lockstep).",
             "N = 21        # number of scan points"]
    for spec in cols:
        lines.append(f"{spec.name} = {sweep(spec, 'N')}        # {note(spec)}")
    slots = ", ".join(spec.name for spec in cols)
    lines.append(f"scan_table = np.column_stack([{slots}])")
    return "\n".join(lines) + "\n"


def _period_display(state: "PulseTableState", index) -> str:
    """A period's human name (its name, else ``period <i>``) -- so a scan target's bare period
    INDEX is never shown to the operator as an opaque number."""
    try:
        i = int(index)
    except (TypeError, ValueError):
        return str(index)
    if 0 <= i < len(state.periods):
        return str(state.periods[i].name).strip() or f"period {i}"
    return f"period {i}"


def scan_target_label(state: "PulseTableState", kind: str, target: str) -> str:
    """The STATE-FUL, NAME-based label for a scannable parameter ``(kind, target)`` -- the ONE
    formatter for callers that HOLD a ``PulseTableState``, turning the opaque ``bus@<index>`` /
    bare-index forms into NAME-based text (``probe duration`` / ``da @ image (DAC, signed LSB)``).
    Shared by the scan-slot axis label, the GUI scan legend, AND ``enumerate_pulse_params`` (so the
    dropdown + the axis read identically).  The COMPLEMENT is the frontend's
    ``pulse_gui.slot_label`` -- the STATE-FREE, INDEX-based label (``Period 3 duration``) for the
    flat row tuples that carry no state.  The raw ``target`` is unchanged (still the parse key);
    this is display only."""
    kind = str(kind)
    target = str(target)
    if kind == "dac" and "@" in target:
        bus, _, idx = target.partition("@")
        return f"{bus} @ {_period_display(state, idx)} (DAC, signed LSB)"
    if kind == "duration":
        return f"{_period_display(state, target)} duration"
    if kind == "delay":
        # A plain CHANNEL delay shows the channel's friendly label; a DAC BUS delay (the bus name
        # is not an individual channel) shows the raw bus name -- matching enumerate_pulse_params.
        name = state.label_for(target) if target in state.channels else target
        return f"{name} delay"
    return f"{kind} {target}"


def evaluate_scan_table_code(code: str, n_slots: int) -> tuple[list[list[float]], tuple[int, int] | None]:
    """Run a scan-table program (see :func:`scan_table_template`) in a small numpy namespace and
    return ``(rows, scan_shape)``: the ``(N_points x n_slots)`` table as a list of rows, and an
    optional 2-D ``scan_shape`` ``(n0, n1)`` the program may assign to declare a GRID scan (so a
    consumer can reshape the per-point result into a 2-D map for a 2D image).  ``scan_shape`` is
    ``None`` for a plain 1-D sweep.

    SECURITY: execs the operator-entered snippet as arbitrary Python -- a LOCAL experiment tool;
    only run programs you wrote or trust.  Raises ``ValueError`` with a fixable message; the
    number of COLUMNS must equal the bound slot count (one column per slot, advanced in lockstep)."""

    import numpy as np

    n = max(1, int(n_slots))
    text = str(code or "").strip()
    if not text:
        raise ValueError("scan program is empty -- assign an (N_points x n_slots) array to 'scan_table'.")
    namespace = {"np": np, "numpy": np, "math": math, "n_slots": n}
    try:
        exec(compile(text, "<scan-table>", "exec"), namespace)   # noqa: S102 - local experiment tool, trusted input
    except Exception as exc:
        raise ValueError(f"scan program did not run: {exc}") from exc
    table = namespace.get("scan_table")
    if table is None:
        raise ValueError("the scan program must assign an (N_points x n_slots) array to a 'scan_table' variable.")
    arr = np.atleast_2d(np.asarray(table, dtype=float))
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError(f"scan_table must be a 2-D (N_points x n_slots) array; got shape {arr.shape}.")
    if arr.shape[1] != n:
        raise ValueError(
            f"scan_table has {arr.shape[1]} column(s) but {n} scan slot(s) are bound -- give ONE column "
            "per slot (the points advance in lockstep; build correlations into the columns).")
    rows = [[float(v) for v in row] for row in arr]
    shape = namespace.get("scan_shape")
    scan_shape: tuple[int, ...] | None = None
    if shape is not None:
        # a GRID scan declares its point-axes shape -- ANY number of nested axes (a 2-level scan is
        # (n0, n1), a 3-level scan (n0, n1, n2), ...): the streamed table stays flat, the shape only
        # tells the display layer how to un-flatten (a 2-D map, a facet grid's outer axis, ...).
        try:
            dims = tuple(int(v) for v in shape)
        except (TypeError, ValueError):
            raise ValueError(f"scan_shape must be a tuple of axis lengths (n0, n1, ...); got {shape!r}.")
        if len(dims) < 2 or any(v < 1 for v in dims):
            raise ValueError(f"scan_shape needs at least two axes of length >= 1; got {shape!r}.")
        if int(np.prod(dims)) != arr.shape[0]:
            raise ValueError(
                f"scan_shape {dims} (= {int(np.prod(dims))} points) does not match the {arr.shape[0]} "
                "scan rows -- a grid scan needs prod(scan_shape) == N_points.")
        scan_shape = dims
    return rows, scan_shape


def quantized_time_ns(
    value_ns: float | str,
    *,
    time_step_ns: float,
    allow_zero: bool,
    allow_negative: bool = False,
) -> float:
    return quantized_time_steps(
        value_ns,
        time_step_ns=time_step_ns,
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
        # _has_context: a NON-EMPTY slot mapping was provided -> the caller is resolving against
        # a known slot set, so an sN missing from it is a typo and must raise.  With NO slots
        # (None) or an EMPTY mapping (e.g. validate after with_slots_resolved cleared the slots,
        # where a leftover delay/duration expression like "s0/2" must still evaluate), keep the
        # lenient 0.0 fallback -- raising there would break legitimate resolve/validate passes.
        self._has_context = bool(self.values)

    def eval(self, text: str) -> float:
        return float(self._visit(ast.parse(_insert_implicit_mul(text), mode="eval").body))

    def _visit(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.values:
                return self.values[node.id]
            if SLOT_VAR_RE.fullmatch(node.id):
                if self._has_context:
                    raise ValueError(f"time expression references unbound scan slot {node.id!r}.")
                return 0.0
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


@dataclass(frozen=True)
class PulseParam:
    """One addressable, software-settable pulse parameter for a per-point scan.

    A measurement scans a parameter by, PER POINT, loading the base template, calling
    :meth:`apply` to set this parameter to the point's value, compiling ``to_sequence()`` and
    firing -- the "non-slot" software scan: no hardware scan slot is needed, so it can sweep a
    period DURATION, a channel DELAY, or a DAC bus level of ANY loaded program (delay and a
    not-yet-slotted duration are settable here even though the streaming scan slots only cover
    duration/dac).

    ``kind``   -- ``"duration"`` | ``"delay"`` | ``"dac"``.
    ``target`` -- duration: the period index as a string; delay: the channel name;
                  dac: ``"<bus>@<period_index>"``.
    ``unit``   -- time unit for duration / delay (``ns``/``us``/``ms``/``s``); IGNORED for dac
                  (DAC values are unitless signed LSB codes, 0 = 0 V).
    """

    kind: str
    target: str
    unit: str = "ns"

    def apply(self, state: "PulseTableState", value: float) -> "PulseTableState":
        """Return a NEW state (the input is never mutated) with this parameter set to ``value``.

        Deep-copies first (the ``set_*`` methods mutate in place) and delegates to the existing
        ``PulseTableState`` setters, so it inherits every guard they carry -- a scan-bound
        duration raises, a DAC value outside the bus signed range raises -- rather than touching
        the periods directly."""
        import copy

        clone = copy.deepcopy(state)
        if self.kind == "duration":
            clone.set_period_duration(int(self.target), float(value), unit=self.unit)
        elif self.kind == "delay":
            clone.set_channel_delay(str(self.target), float(value), unit=self.unit)
        elif self.kind == "dac":
            bus, _, period = str(self.target).partition("@")
            clone.set_analog_bus_mode(int(period), bus, "edge", value=int(round(float(value))))
        else:
            raise ValueError(f"unknown PulseParam kind {self.kind!r} (expected duration/delay/dac).")
        return clone


def period_index_by_name(state: "PulseTableState", name: str) -> int | None:
    """Index of the first period whose name matches ``name`` (case-insensitive, stripped), or
    ``None`` if no period carries that name.  The single source for name->index resolution
    (the scan-target enumeration + any name-based period lookup use it)."""

    key = str(name).strip().lower()
    return next((i for i, p in enumerate(state.periods)
                 if str(p.name).strip().lower() == key), None)


def enumerate_pulse_params(state: "PulseTableState") -> list[tuple[str, str, str]]:
    """Every software-scannable parameter of ``state`` as ``(kind, target, label)`` triples --
    the ONE source the GUI scan-target dropdown and a notebook both read.  Covers every period
    DURATION, every channel DELAY, and every DAC bus level per period; a duration already bound
    to a scan slot is flagged in its label (setting it raises until the slot is unbound)."""

    # Labels come from the ONE formatter (scan_target_label); enumerate only adds the
    # ' (scan-bound)' flag (the duration-already-bound hint), AROUND the call (#B3).
    out: list[tuple[str, str, str]] = []
    for i, _period in enumerate(state.periods):
        bound = state.slot_index_for("duration", str(i)) is not None
        out.append(("duration", str(i),
                    scan_target_label(state, "duration", str(i)) + (" (scan-bound)" if bound else "")))
    # Delays are API-settable (never scannable); a DAC BUS owns a delay just like a
    # plain channel (it fans out to its members), so both are listed -- buses first so
    # the bus-delay slot is as discoverable as a TTL channel's.
    bus_channels = state.bus_channels(min_width=1)
    bus_members = {channel for members in bus_channels.values() for channel in members}
    for bus in bus_channels:
        out.append(("delay", str(bus), scan_target_label(state, "delay", str(bus))))
    for channel in state.channels:
        if channel in bus_members:
            continue   # its delay is exposed via the owning bus's slot, not per-member
        out.append(("delay", str(channel), scan_target_label(state, "delay", str(channel))))
    for bus in bus_channels:
        for i, _period in enumerate(state.periods):
            out.append(("dac", f"{bus}@{i}", scan_target_label(state, "dac", f"{bus}@{i}")))
    return out


__all__ = [
    "ANALOG_BUS_MODES",
    "SCAN_SLOT_KINDS",
    "PulseParam",
    "PulsePeriod",
    "PulseTableState",
    "ScanSlot",
    "enumerate_pulse_params",
    "period_index_by_name",
    "affine_coeffs",
    "default_pulse_name",
    "default_periods",
    "default_visible_channels",
    "eval_time_expr",
    "evaluate_scan_table_code",
    "infer_bus_channels",
    "load_scan_table",
    "positive_time_step_ns",
    "quantized_time_ns",
    "quantized_time_steps",
    "scan_table_template",
    "scan_target_label",
    "slot_var",
    "snap_scan_table",
]
