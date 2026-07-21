"""Turn what the editor holds into what the pulse pipeline runs.

The editor's ``PulseTableState`` and the pipeline's ``PulseDocument`` describe the
same pulse programme in two shapes, and nothing connected them -- which is why the
editor could author a programme it had no way to execute.  This module is that
connection, and it lives OUTSIDE ``zlc_neutral_atom.timing`` on purpose: the two
representations inside that package are forbidden to import each other, and the
translation must not be the thing that marries them.

Two differences are real, not cosmetic, and are where the work is:

* **DAC values.**  The editor encodes a bus's value into the raw TTL bit vector it
  keeps per period; a document keeps every DAC lane's bit at 0 and states the value
  once, as an ``AnalogStep``.  Both spell the value in the SIGNED user layer (0 = 0 V),
  so no re-encoding happens here -- only relocation.
* **Scanned durations.**  The editor stores a duration as an expression over slot
  names; a document stores a concrete duration plus a typed field binding, with the
  scan's own values in the frozen table.  The nominal slot values are substituted to
  get the concrete duration, which is exactly what the editor shows the operator.

Periods are addressed positionally in the editor and by identity in a document, so
this is also where a period acquires a stable id.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from zlc_neutral_atom.timing.pulse_table import PulseTableState
from zlc_neutral_atom.timing.ports import PORT_DAC
from zlc_pulse.authoring import attach_scan_recipe, freeze_scan_table
from zlc_pulse.document import (
    FIELD_DAC,
    FIELD_DELAY,
    FIELD_DURATION,
    AnalogStep,
    ApiParameter,
    OutputDelay,
    PulseDocument,
    PulseFieldRef,
    PulsePeriod,
    RepeatRegion,
    ScanParameter,
)
from zlc_pulse.target import PulseTarget

__all__ = ["pulse_document_from_table"]

#: Nanoseconds per unit the editor may express a duration or delay in.
_UNIT_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
#: The editor's word for "this bus keeps whatever it had"; a document says the same
#: thing by leaving the period out.
_HOLD = "hold"
#: The scan recipe is a Python snippet the operator wrote in the Scan tab.
_RECIPE_LANGUAGE = "python"


def _period_id(index: int) -> str:
    """Positional in the editor, identified in a document."""

    return f"p{index + 1}"


def _field_ref(kind: str, target: str, *, period_ids: tuple[str, ...]) -> PulseFieldRef:
    """Resolve an editor slot target onto a typed document field reference.

    The editor writes a duration target as a period INDEX and a DAC target as
    ``"<bus>@<period index>"``; a delay target is the channel itself.
    """

    if kind == FIELD_DELAY:
        return PulseFieldRef(FIELD_DELAY, port=str(target))
    if kind == FIELD_DAC:
        bus, _, index_text = str(target).partition("@")
        if not _:
            raise ValueError(f"DAC slot target {target!r} must look like '<bus>@<period index>'")
        return PulseFieldRef(FIELD_DAC, period_id=period_ids[int(index_text)], port=bus)
    if kind == FIELD_DURATION:
        return PulseFieldRef(FIELD_DURATION, period_id=period_ids[int(target)])
    raise ValueError(f"unsupported slot kind {kind!r}")


def _duration_and_unit(period, *, slots: Mapping[str, float]) -> tuple[float, str]:
    """The concrete duration, in the unit it is honestly expressed in.

    A scanned duration is an EXPRESSION over slot variables, and the editor then marks
    the period's unit field as a note to itself rather than a unit ("str (ns)").  The
    expression's value is nanoseconds, so that is what a document must say; taking the
    marker for a unit would silently scale the pulse.
    """

    unit = str(period.unit or "ns")
    nanoseconds = period.duration_ns(slots=dict(slots))
    if unit not in _UNIT_NS:
        return nanoseconds, "ns"
    return nanoseconds / _UNIT_NS[unit], unit


def pulse_document_from_table(state: PulseTableState, target: PulseTarget) -> PulseDocument:
    """Translate an editor state onto ``target``.

    ``target`` is supplied by the caller because it carries hardware facts the editor's
    catalog does not (bus width, encoding, safe level): it comes from the board, or from
    the sequencer the document is destined for.
    """

    if not isinstance(state, PulseTableState):
        raise TypeError("state must be PulseTableState")
    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")

    period_ids = tuple(_period_id(index) for index in range(len(state.periods)))
    # A duration expression names its slot by the COMPILER column (``s0``), while the slot
    # itself carries a semantic name (``hold_time``).  Both are offered so either spelling
    # resolves; feeding only the semantic name leaves ``s0`` unbound.
    nominal: dict[str, float] = {}
    for column, slot in zip(state.compiler_scan_vars, state.scan_slots):
        nominal[column] = slot.nominal
        nominal[slot.name] = slot.nominal
    dac_lanes = {lane for port in target.ports if port.kind == PORT_DAC for lane in port.lanes}

    periods: list[PulsePeriod] = []
    for index, period in enumerate(state.periods):
        # A DAC lane's bit belongs to the bus's AnalogStep, not to the digital vector.
        states = tuple(
            0 if lane in dac_lanes else int(bit)
            for lane, bit in zip(target.raw_lanes, period.states)
        )
        analog_steps = []
        for bus, entries in state.analog_bus_modes.items():
            if index >= len(entries):
                continue
            entry = entries[index]
            mode = str(entry.get("mode", _HOLD))
            if mode == _HOLD or entry.get("value") is None:
                continue          # a document says "hold" by saying nothing
            analog_steps.append(AnalogStep(bus, mode, int(entry["value"])))
        duration, unit = _duration_and_unit(period, slots=nominal)
        periods.append(PulsePeriod(
            period_id=period_ids[index],
            duration=duration,
            unit=unit,
            name=str(period.name or ""),
            states=states,
            analog_steps=tuple(sorted(analog_steps, key=lambda step: step.port)),
        ))

    scan_parameters = tuple(
        ScanParameter(slot.name, _field_ref(slot.kind, slot.target, period_ids=period_ids),
                      slot.label or slot.name, slot.unit)
        for slot in state.scan_slots
    )
    api_parameters = tuple(
        ApiParameter(slot.name, _field_ref(slot.kind, slot.target, period_ids=period_ids),
                     slot.unit)
        for slot in state.api_slots
    )
    delays = tuple(
        OutputDelay(port, value, str(state.delay_units.get(port, "ns")))
        for port, value in sorted(state.delays.items())
    )

    repeat = None
    if (state.repeat_start is not None and state.repeat_end is not None
            and int(state.repeat_count) >= 2):
        repeat = RepeatRegion(period_ids[int(state.repeat_start)],
                              period_ids[int(state.repeat_end)],
                              int(state.repeat_count))

    document = PulseDocument(
        name=state.name,
        target=target,
        time_step_ns=state.time_step_ns,
        periods=tuple(periods),
        scan_parameters=scan_parameters,
        api_parameters=api_parameters,
        visible_ports=tuple(state.visible_ports),
        delays=delays,
        repeat=repeat,
    )

    # The frozen table and its recipe are NOT assembled here: freezing normalises every
    # value onto the field it feeds and stamps a fingerprint, and attaching a recipe proves
    # the source reproduces that exact table.  Both checks belong to the owner of the
    # format, so the bridge hands its rows over and lets that owner do the work.
    columns = tuple(slot.name for slot in state.scan_slots)
    rows = tuple(tuple(row) for row in state.scan_table)
    if columns and rows:
        table, _report = freeze_scan_table(document, columns, rows)
        document = replace(document, scan_table=table)
        source = str(state.scan_code or "")
        if source and not source.isspace():     # the snippet is kept verbatim, never trimmed
            document = attach_scan_recipe(
                document,
                source=state.scan_code,
                # The editor's table IS this snippet's output -- pressing Run in the Scan
                # tab is what produced the rows now being frozen.
                generated_columns={name: [row[index] for row in rows]
                                   for index, name in enumerate(columns)},
                language=_RECIPE_LANGUAGE,
            )
    return document
