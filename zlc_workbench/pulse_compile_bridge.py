"""Migration bridge from current pulse values to the proven installed compiler.

The bridge is composition-private: zlc_pulse owns PulseDocument/TargetIR and has
no dependency on the historical neutral-atom pulse implementation.  Until the
compiler algorithms are moved behind the pulse boundary, this one adapter
translates values, invokes the existing compiler/validator, and immediately
discards every historical object.
"""

from __future__ import annotations

from enum import Enum

from Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer import (
    validate_pulse_streamer_program,
)
from Zou_lab_control.neutral_atom.devices.sequencer import (
    compile_pulse_table_runtime_program,
    compile_pulse_table_scan_runtime_program,
)
from Zou_lab_control.neutral_atom.ports import PortCatalog, PortSpec
from Zou_lab_control.neutral_atom.timing.pulse_table import (
    ApiSlot as LegacyApiSlot,
    PulsePeriod as LegacyPulsePeriod,
    PulseTableState,
    ScanSlot as LegacyScanSlot,
)
from zlc_pulse import (
    PulseDocument,
    PulseTarget,
    TargetBusDelay,
    TargetBusSegment,
    TargetIR,
)


class PulseExecutionForm(str, Enum):
    STATIC_ONCE = "STATIC_ONCE"
    STATIC_REFERENCE_POINT = "STATIC_REFERENCE_POINT"
    CONTINUOUS_MONITOR = "CONTINUOUS_MONITOR"
    AUTONOMOUS_SCAN_ONCE = "AUTONOMOUS_SCAN_ONCE"


def compile_pulse_document(
    document: PulseDocument,
    *,
    clock_hz: float,
    execution_form: PulseExecutionForm,
    live_target: PulseTarget | None = None,
) -> TargetIR:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(execution_form, PulseExecutionForm):
        raise TypeError("execution_form must be PulseExecutionForm")
    if live_target is not None:
        if not isinstance(live_target, PulseTarget):
            raise TypeError("live_target must be PulseTarget")
        if live_target.abi_fingerprint != document.target.abi_fingerprint:
            raise ValueError("pulse document target ABI differs from live target")
    state, catalog = _legacy_compile_input(document)
    if execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        if not document.scan_slots or not document.scan_table:
            raise ValueError("AUTONOMOUS_SCAN_ONCE requires scan slots and points")
        if document.scan_repeats != 0:
            raise ValueError(
                "finite autonomous scan expands repeats in its frozen plan; "
                "legacy cursor-wrap scan_repeats is forbidden"
            )
        program = compile_pulse_table_scan_runtime_program(
            state,
            clock_hz=clock_hz,
            repeat_forever=False,
            port_catalog=catalog,
        )
    else:
        reference = execution_form is PulseExecutionForm.STATIC_REFERENCE_POINT
        if (document.scan_slots or document.scan_table) and not reference:
            raise ValueError("static execution cannot silently ignore a scan definition")
        if reference and not document.scan_slots:
            raise ValueError("STATIC_REFERENCE_POINT requires at least one scan slot")
        program = compile_pulse_table_runtime_program(
            state,
            clock_hz=clock_hz,
            repeat_forever=execution_form is PulseExecutionForm.CONTINUOUS_MONITOR,
            port_catalog=catalog,
        )
    validate_pulse_streamer_program(program)
    return _target_ir(program, document.target)


def _legacy_compile_input(document: PulseDocument) -> tuple[PulseTableState, PortCatalog]:
    ports = tuple(
        PortSpec(
            key=port.key,
            kind=port.kind,
            lanes=port.lanes,
            label=port.label,
            bus_index=port.bus_index,
            width=port.width,
            encoding=port.encoding,
            safe_value=port.safe_value,
            latch_clock=port.latch_clock,
        )
        for port in document.target.ports
    )
    catalog = PortCatalog(document.target.raw_lanes, ports)
    state = PulseTableState(
        name=document.name,
        port_catalog=catalog,
        periods=[
            LegacyPulsePeriod(period.duration, period.states, unit=period.unit, name=period.name)
            for period in document.periods
        ],
        scan_slots=[
            LegacyScanSlot(
                slot.kind,
                slot.target,
                label=slot.label,
                unit=slot.unit,
                nominal=slot.nominal,
                name=slot.name,
            )
            for slot in document.scan_slots
        ],
        scan_table=document.scan_table,
        scan_code=document.scan_code,
        api_slots=[
            LegacyApiSlot(slot.name, slot.kind, slot.target, slot.unit)
            for slot in document.api_slots
        ],
        time_step_ns=document.time_step_ns,
        visible_ports=document.visible_ports,
        analog_bus_modes={
            key: [
                {"mode": step.mode, "value": step.value}
                for step in steps
            ]
            for key, steps in document.analog_bus_programs
        },
        delays=dict(document.delays),
        delay_units=dict(document.delay_units),
        repeat_start=document.repeat_start,
        repeat_end=document.repeat_end,
        repeat_count=document.repeat_count,
        repeat_forever=document.repeat_forever,
        scan_repeats=document.scan_repeats,
    )
    return state, catalog


def _target_ir(program, target: PulseTarget) -> TargetIR:
    if int(getattr(program, "repeat_from_index", 0)) != 0:
        raise ValueError(
            "installed compiler produced unsupported nonzero repeat_from_index"
        )
    ticks = tuple(int(value) for value in program.ticks)
    slot_kinds = tuple(program.slot_kinds or ())
    slot_count = len(slot_kinds)
    tick_coeffs = tuple(
        tuple(int(value) for value in row)
        for row in (
            program.tick_slot_coeffs
            or tuple((0,) * slot_count for _ in ticks)
        )
    )
    loop_coeffs = tuple(
        int(value)
        for value in (program.loop_end_slot_coeffs or (0,) * slot_count)
    )
    bus_segments = tuple(
        TargetBusSegment(
            int(segment.bus_index),
            str(segment.bus_name),
            int(segment.start_tick),
            int(segment.stop_tick),
            int(segment.start_value),
            int(segment.stop_value),
            str(segment.mode),
            int(segment.value_select),
            int(segment.stop_value_select),
            tuple(int(value) for value in (segment.start_tick_coeffs or (0,) * slot_count)),
            tuple(int(value) for value in (segment.stop_tick_coeffs or (0,) * slot_count)),
        )
        for segment in (program.bus_segments or ())
    )
    channel_delays = tuple(
        int(value)
        for value in (
            program.channel_delays
            or (0,) * len(program.channels)
        )
    )
    return TargetIR(
        float(program.clock_hz),
        target.abi_fingerprint,
        tuple(str(value) for value in program.channels),
        ticks,
        tuple(int(value) for value in program.masks),
        float(program.duration),
        bool(program.repeat_forever),
        int(program.loop_start_index),
        int(program.loop_end_tick),
        int(program.loop_count),
        slot_kinds,
        loop_coeffs,
        tick_coeffs,
        tuple(tuple(int(value) for value in row) for row in (program.scan_points or ())),
        tuple(float(value) for value in (program.scan_point_durations or ())),
        int(program.scan_coeff_frac_bits) if slot_count else 0,
        int(program.scan_repeats) if slot_count else 0,
        tuple(str(value) for value in (program.bus_names or ())),
        bus_segments,
        tuple(
            TargetBusDelay(int(value.bus_index), int(value.delay))
            for value in (program.bus_delays or ())
        ),
        channel_delays,
        int(program.clk_enable),
    )


__all__ = ["PulseExecutionForm", "compile_pulse_document"]
