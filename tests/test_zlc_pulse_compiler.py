from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zlc_pulse import (
    COMPILER_ID,
    COMPILER_VERSION,
    FIELD_DURATION,
    OutputDelay,
    PORT_DAC,
    PORT_DIGITAL,
    PulseExecutionForm,
    PulseFieldRef,
    RepeatRegion,
    ScanParameter,
    compile_pulse_artifact,
    compile_pulse_document,
    freeze_scan_table,
    load_pulse_document,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "form", "expected_ir_digest"),
    (
        ("camera_imaging_address_switch.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "1824c5f4ea06c6c8a299d8e33da5d60071e2b7998155892aa05e4604808eb5fd"),
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE, "72c6b9a176b17c8882689e19fea319156c67d3398ccd16f30f9a30ed4a91f073"),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "2684cbf2dff424bd24595d6e688d74418ef8c3037aca7be3f5f419864d031c00"),
        ("probe_template.json", PulseExecutionForm.STATIC_ONCE, "d60570632f4c840e49d0e9270a10ee7fb073667c6519dc9aef1d8dfdf37b7e80"),
        ("release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "ec0b8ae0ad75e12066e1e4c519140d370e5091fbdb591bce53ea5415c7ab036d"),
    ),
)
def test_native_compiler_preserves_every_tracked_physical_golden(
    filename, form, expected_ir_digest
):
    document = load_pulse_document(ROOT / "pulses" / filename)

    actual = compile_pulse_document(document, clock_hz=50e6, execution_form=form)

    assert actual.fingerprint == expected_ir_digest


def test_native_compiler_preserves_compact_and_delay_unrolled_repeat_semantics():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    active_port = next(
        port
        for port in document.target.ports
        if port.kind == PORT_DIGITAL
        and any(
            period.states[document.target.raw_lanes.index(port.lanes[0])]
            for period in document.periods
        )
    )
    compact = replace(document, repeat=RepeatRegion("p2", "p4", 3))
    delayed = replace(compact, delays=(OutputDelay(active_port.key, -40, "ns"),))

    compact_ir = compile_pulse_document(
        compact,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
    delayed_ir = compile_pulse_document(
        delayed,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )

    assert compact_ir.fingerprint == "46111a335a8a8d63e506a025dfde9101f23c8b5ec983a50fb75357cc2169521d"
    assert delayed_ir.fingerprint == "8eadcf2f08a9ccd7de4095d4bef021433a1f837f3acb47004e19fdb96bfc16e5"
    assert compact_ir.loop_count == 3
    assert delayed_ir.loop_count == 1


def test_native_compiler_folds_digital_and_dac_delays_in_one_schedule():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    active_bus_keys = {
        step.port for period in document.periods for step in period.analog_steps
    }
    bus = next(
        port
        for port in document.target.ports
        if port.kind == PORT_DAC and port.key in active_bus_keys
    )
    digital = next(
        port
        for port in document.target.ports
        if port.kind == PORT_DIGITAL
        and any(
            period.states[document.target.raw_lanes.index(port.lanes[0])]
            for period in document.periods
        )
    )
    delayed = replace(
        document,
        delays=(
            OutputDelay(bus.key, 40, "ns"),
            OutputDelay(digital.key, 100, "ns"),
        ),
    )

    actual = compile_pulse_document(
        delayed,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )

    assert any(actual.channel_delays)
    assert any(delay.delay_ticks for delay in actual.bus_delays)


def test_scan_duration_includes_every_compact_inner_repeat_iteration():
    base = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    parameter = ScanParameter(
        "gap_duration",
        PulseFieldRef(FIELD_DURATION, "p3"),
        "gap duration",
        "s",
    )
    document = replace(
        base,
        scan_parameters=(parameter,),
        repeat=RepeatRegion("p2", "p4", 3),
    )
    table, _report = freeze_scan_table(
        document,
        (parameter.parameter_id,),
        ((0.0001,), (0.0002,)),
    )
    document = replace(document, scan_table=table)

    actual = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    )

    assert actual.loop_count == 3
    assert actual.scan_point_durations == pytest.approx((0.0974, 0.0977))
    assert actual.duration_seconds == pytest.approx(0.1951)


def test_scan_rejects_dac_segments_inside_compact_inner_repeat():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    unsupported = replace(document, repeat=RepeatRegion("p2", "p2", 2))

    with pytest.raises(ValueError, match="DAC bus segments.*inner repeat"):
        compile_pulse_document(
            unsupported,
            clock_hz=50e6,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        )


def test_artifact_uses_v2_compiler_identity_and_one_geometry_for_ir_and_wire():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    trigger = next(
        port.lanes[0]
        for port in document.target.ports
        if port.kind == PORT_DIGITAL
        and any(
            period.states[document.target.raw_lanes.index(port.lanes[0])]
            for period in document.periods
        )
    )

    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channels=(trigger,),
    )

    assert artifact.compiler_id == COMPILER_ID
    assert artifact.compiler_version == COMPILER_VERSION == "2"
    assert artifact.target_ir.scan_coeff_frac_bits == 8
    assert artifact.wire_image.source_ir_digest == artifact.target_ir.fingerprint


def test_compiler_rejects_scan_loss_and_clock_grid_drift():
    scan = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    with pytest.raises(ValueError, match="silently ignore"):
        compile_pulse_document(
            scan,
            clock_hz=50e6,
            execution_form=PulseExecutionForm.STATIC_ONCE,
        )
    with pytest.raises(ValueError, match="clock grid differs"):
        compile_pulse_document(
            scan,
            clock_hz=40e6,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        )
    assert not hasattr(scan, "scan_repeats")
    assert not hasattr(scan, "repeat_forever")
