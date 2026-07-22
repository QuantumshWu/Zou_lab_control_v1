from __future__ import annotations

from dataclasses import replace
from inspect import signature

from fpga.pulse_streamer.host.engine_model import rtl_bus_segment_delay_mirror

from zlc_pulse import (
    DAC_OFFSET_BINARY,
    FIELD_DAC,
    FIELD_DURATION,
    PORT_CLOCK,
    PORT_DAC,
    PORT_DIGITAL,
    AnalogStep,
    FrozenScanTable,
    OutputDelay,
    PulseDocument,
    PulseExecutionForm,
    PulseFieldRef,
    PulsePeriod,
    PulsePortSpec,
    PulseTarget,
    RepeatRegion,
    ScanParameter,
    build_pulse_timeline,
    compile_pulse_artifact,
    compile_pulse_document,
)


def _target() -> PulseTarget:
    return PulseTarget(
        (
            "live_lane",
            "off_lane",
            "dac0_bit0",
            "dac0_bit1",
            "dac0_bit2",
            "dac0_clock_lane",
            "dac1_bit0",
            "dac1_bit1",
            "dac1_bit2",
            "dac1_clock_lane",
        ),
        (
            PulsePortSpec(
                "ttl_live", PORT_DIGITAL, ("live_lane",), "Live TTL",
                None, 1, "binary", 0, None,
            ),
            PulsePortSpec(
                "ttl_off", PORT_DIGITAL, ("off_lane",), "Off TTL",
                None, 1, "binary", 0, None,
            ),
            PulsePortSpec(
                "dac_live", PORT_DAC,
                ("dac0_bit0", "dac0_bit1", "dac0_bit2"), "Live DAC",
                0, 3, DAC_OFFSET_BINARY, 4, "dac0_clock",
            ),
            PulsePortSpec(
                "dac0_clock", PORT_CLOCK, ("dac0_clock_lane",), "DAC 0 clock",
                None, 1, "binary", 0, None,
            ),
            PulsePortSpec(
                "dac_off", PORT_DAC,
                ("dac1_bit0", "dac1_bit1", "dac1_bit2"), "Off DAC",
                1, 3, DAC_OFFSET_BINARY, 4, "dac1_clock",
            ),
            PulsePortSpec(
                "dac1_clock", PORT_CLOCK, ("dac1_clock_lane",), "DAC 1 clock",
                None, 1, "binary", 0, None,
            ),
        ),
    )


def _document() -> PulseDocument:
    target = _target()
    all_low = (0,) * len(target.raw_lanes)
    live = (1, *all_low[1:])
    return PulseDocument(
        "authored timeline",
        target,
        10.0,
        (
            PulsePeriod(
                "p1", 40, "ns", "one", live,
                (AnalogStep("dac_live", "edge", -3),),
            ),
            PulsePeriod("p2", 50, "ns", "two", all_low),
            PulsePeriod(
                "p3", 60, "ns", "three", all_low,
                (AnalogStep("dac_live", "ramp", 3),),
            ),
        ),
        # Preview row retention is target-owned, not coupled to the Edit tab's
        # currently selected subset.
        visible_ports=("ttl_live",),
        delays=(
            OutputDelay("ttl_live", 50, "ns"),
            OutputDelay("dac_live", 20, "ns"),
        ),
        repeat=RepeatRegion("p1", "p2", 3),
    )


def _timeline(document: PulseDocument):
    artifact = compile_pulse_artifact(
        document,
        clock_hz=100e6,
        execution_form=(
            PulseExecutionForm.STATIC_REFERENCE_POINT
            if document.scan_parameters
            else PulseExecutionForm.STATIC_ONCE
        ),
    )
    return build_pulse_timeline(
        document,
        artifact,
        reference_label=(
            "nominal scan/API reference"
            if document.scan_parameters
            else "compiled static pulse"
        ),
    )


def _row_values(row, duration: int) -> tuple[int, ...]:
    return tuple(
        next(
            segment.start_value
            for segment in row.segments
            if segment.start_tick <= tick < segment.stop_tick
        )
        for tick in range(duration)
    )


def test_timeline_keeps_one_authored_period_table_and_every_logical_target_row():
    document = _document()
    timeline = _timeline(document)

    # The execution artifact repeats p1/p2 three times, but the formal editor
    # preview shows the authored p1/p2/p3 table once.
    assert timeline.logical_duration_ticks == 15
    # The last TTL high ended in p1.  The physical axis still includes the
    # compiler-owned five-tick drain after authored DONE; it is not inferred
    # only from the final visible high interval.
    assert timeline.duration_ticks == 20
    periods = [item for item in timeline.annotations if item.kind == "period"]
    assert [(item.start_tick, item.stop_tick, item.label) for item in periods] == [
        (0, 4, "one"),
        (4, 9, "two"),
        (9, 15, "three"),
    ]
    repeat = [item for item in timeline.annotations if item.kind == "repeat"]
    assert [(item.start_tick, item.stop_tick, item.label) for item in repeat] == [
        (0, 9, "×3")
    ]

    assert [row.row_id for row in timeline.rows] == [
        "ttl_live", "ttl_off", "dac_live", "dac_off",
    ]
    assert [row.port_kind for row in timeline.rows] == [
        PORT_DIGITAL, PORT_DIGITAL, PORT_DAC, PORT_DAC,
    ]
    assert [row.active for row in timeline.rows] == [True, False, True, False]
    assert all(row.segments[-1].stop_tick == timeline.duration_ticks for row in timeline.rows)
    assert all(
        segment.start_value == segment.stop_value
        for row in timeline.rows
        for segment in row.segments
    )


def test_dac_timeline_is_the_cycle_exact_delayed_hardware_staircase():
    document = _document()
    timeline = _timeline(document)
    authored_ir = compile_pulse_document(
        replace(document, repeat=None),
        clock_hz=100e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
    delay = next(item.delay_ticks for item in authored_ir.bus_delays if item.bus_index == 0)
    hardware_codes = rtl_bus_segment_delay_mirror(
        authored_ir,
        0,
        delay,
        timeline.duration_ticks,
        bus_width=3,
    )
    row = next(item for item in timeline.rows if item.row_id == "dac_live")

    assert _row_values(row, timeline.duration_ticks) == tuple(
        encoded - 4 for encoded in hardware_codes
    )


def test_scan_overlays_are_typed_and_numbered_in_frozen_hardware_column_order():
    document = _document()
    duration = ScanParameter(
        "duration_slot",
        PulseFieldRef(FIELD_DURATION, "p2"),
        "Duration",
        "ns",
    )
    dac = ScanParameter(
        "dac_slot",
        PulseFieldRef(FIELD_DAC, "p1", "dac_live"),
        "DAC",
        "value",
    )
    scanned = replace(
        document,
        scan_parameters=(duration, dac),
        scan_table=FrozenScanTable(
            (dac.parameter_id, duration.parameter_id),
            ((-3, 50), (2, 70)),
        ),
    )

    timeline = _timeline(scanned)
    overlays = [item for item in timeline.annotations if item.number is not None]

    assert [item.kind for item in overlays] == ["scan-dac", "scan-duration"]
    assert [item.number for item in overlays] == [1, 2]
    assert overlays[0].parameter_id == "dac_slot"
    assert overlays[0].row_id == "dac_live"
    assert overlays[0].value == -3
    assert (overlays[0].start_tick, overlays[0].stop_tick) == (0, 4)
    assert overlays[1].parameter_id == "duration_slot"
    assert overlays[1].row_id is None and overlays[1].value is None
    assert (overlays[1].start_tick, overlays[1].stop_tick) == (4, 9)


def test_timeline_fingerprint_covers_every_render_input():
    timeline = _timeline(_document())
    assert _timeline(_document()).fingerprint == timeline.fingerprint
    assert "max_timeline_items" not in signature(build_pulse_timeline).parameters

    changed_row = replace(timeline.rows[0], active=not timeline.rows[0].active)
    changed_annotation = replace(timeline.annotations[0], label="renamed period")
    variants = (
        replace(timeline, title="renamed timeline"),
        replace(timeline, reference_label="different reference"),
        replace(timeline, rows=(changed_row, *timeline.rows[1:])),
        replace(timeline, annotations=(changed_annotation, *timeline.annotations[1:])),
    )
    assert all(item.fingerprint != timeline.fingerprint for item in variants)
