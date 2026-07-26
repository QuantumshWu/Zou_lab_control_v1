"""Readout applicability is derived from physical pulse edges, not shape guesses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    PointLayout,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    ValidityContract,
    ValueSchema,
)
from zlc_neutral_atom.logic_nodes.readout.physical_context import (
    BusReadoutTrace,
    DigitalReadoutTrace,
    derive_readout_physical_context,
    readout_physical_context_from_tree,
    readout_physical_context_to_tree,
)
from zlc_neutral_atom.timing.capture_plan import compile_capture_cell_plan
from zlc_neutral_atom.timing.lineage import PulseCaptureBinding
from zlc_pulse import (
    CompiledPulseArtifact,
    PulseExecutionForm,
    TargetBusDelay,
    TargetBusSegment,
    TargetIR,
    build_digital_trigger_schedules,
    compile_pulse_artifact,
    load_pulse_document,
    pack_target_ir,
)


ROOT = Path(__file__).parents[1]
CLOCK_HZ = 50e6
IMAGING_TEMPLATE = ROOT / "pulses" / "imaging_template.json"


def _axis(name: str, role: str, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def _artifact(
    *,
    channels: tuple[str, ...],
    ticks: tuple[int, ...],
    masks: tuple[int, ...],
    logical_outputs: tuple[tuple[str, str], ...],
    bus_names: tuple[str, ...] = (),
    bus_safe_values: tuple[int, ...] = (),
    bus_segments: tuple[TargetBusSegment, ...] = (),
    bus_delays: tuple[TargetBusDelay, ...] = (),
    slot_kinds: tuple[str, ...] = (),
    scan_points: tuple[tuple[int, ...], ...] = (),
    target_abi_fingerprint: str = "a" * 64,
) -> CompiledPulseArtifact:
    slot_count = len(slot_kinds)
    coefficients = tuple((0,) * slot_count for _ in ticks)
    point_duration = ticks[-1] / CLOCK_HZ
    ir = TargetIR(
        clock_hz=CLOCK_HZ,
        target_abi_fingerprint=target_abi_fingerprint,
        channels=channels,
        ticks=ticks,
        masks=masks,
        duration_seconds=point_duration * max(1, len(scan_points)),
        repeat_forever=False,
        loop_start_index=0,
        loop_end_tick=ticks[-1],
        loop_count=1,
        slot_kinds=slot_kinds,
        loop_end_slot_coeffs=(0,) * slot_count,
        tick_slot_coeffs=coefficients,
        scan_points=scan_points,
        scan_point_durations=(point_duration,) * len(scan_points),
        scan_coeff_frac_bits=8 if scan_points else 0,
        bus_names=bus_names,
        bus_segments=bus_segments,
        bus_delays=bus_delays,
        channel_delays=(0,) * len(channels),
        logical_digital_outputs=logical_outputs,
        bus_safe_values=bus_safe_values,
    )
    execution_form = (
        PulseExecutionForm.AUTONOMOUS_SCAN_ONCE
        if scan_points
        else PulseExecutionForm.STATIC_ONCE
    )
    trigger_lane = dict(logical_outputs)["trigger"]
    return CompiledPulseArtifact(
        "b" * 64,
        "physical-context-test",
        execution_form,
        ir,
        pack_target_ir(ir),
        build_digital_trigger_schedules(ir, (trigger_lane,)),
    )


def _cell_plan(
    artifact: CompiledPulseArtifact,
    *,
    repeats: int = 1,
    readout_events: int = 1,
):
    event_axis = _axis("capture.event", READOUT_EVENT, readout_events)
    point_count = max(1, len(artifact.target_ir.scan_points))
    if artifact.target_ir.scan_points:
        scan_axis = _axis("capture.scan", SCAN_POINT, point_count)
        point_axes = (scan_axis, event_axis)
        dataset_layout = PointLayout.rect_c((point_count, readout_events))
        scan_layout = PointLayout.rect_c((point_count,))
    else:
        point_axes = (event_axis,)
        dataset_layout = PointLayout.rect_c((readout_events,))
        scan_layout = PointLayout.rect_c(())
    schema = DatasetSchema(
        _axis("capture.repeat", REPEAT, repeats),
        point_axes,
        dataset_layout,
        ValueSchema.scalar(np.dtype("<u2"), "count"),
    )
    trigger_lane = artifact.trigger_schedules[0].channel
    return compile_capture_cell_plan(
        artifact,
        trigger_lane,
        schema,
        readout_event_axis_id=event_axis.axis_id,
        scan_point_layout=scan_layout,
    )


def _derive(
    artifact: CompiledPulseArtifact,
    *,
    offset_ticks: float,
    exposure_ticks: float,
    repeats: int = 1,
):
    plan = _cell_plan(artifact, repeats=repeats)
    return derive_readout_physical_context(
        PulseCaptureBinding(artifact, plan.trigger_channel, plan),
        readout_event_index=0,
        integration_start_offset_seconds=offset_ticks / CLOCK_HZ,
        integration_seconds=exposure_ticks / CLOCK_HZ,
    )


def _probe_artifact(
    *,
    probe_stop: int = 40,
    aux_start: int = 80,
    aux_stop: int = 90,
    target_abi_fingerprint: str = "a" * 64,
) -> CompiledPulseArtifact:
    ticks = tuple(sorted({0, 10, 20, probe_stop, 60, aux_start, aux_stop, 100}))

    def mask(tick: int) -> int:
        return (
            int(aux_start <= tick < aux_stop)
            | (int(20 <= tick < probe_stop) << 1)
            | (int(10 <= tick < 60) << 2)
        )

    return _artifact(
        channels=("aux-lane", "probe-lane", "trigger-lane"),
        ticks=ticks,
        masks=tuple(mask(tick) for tick in ticks),
        logical_outputs=(
            ("aux", "aux-lane"),
            ("probe", "probe-lane"),
            ("trigger", "trigger-lane"),
        ),
        target_abi_fingerprint=target_abi_fingerprint,
    )


def test_fractional_offset_and_half_open_boundaries_are_exact() -> None:
    artifact = _probe_artifact()

    exact = _derive(artifact, offset_ticks=10.0, exposure_ticks=20.0)
    assert exact.digital == (DigitalReadoutTrace("aux", False), DigitalReadoutTrace("probe", True))

    fractional = _derive(artifact, offset_ticks=9.5, exposure_ticks=20.5)
    assert fractional.integration_start_offset_seconds == 9.5 / CLOCK_HZ
    assert fractional.digital == (
        DigitalReadoutTrace("aux", False),
        DigitalReadoutTrace("probe", False, ((10, True),)),
    )


def test_only_waveform_inside_the_integration_window_changes_context() -> None:
    baseline = _derive(
        _probe_artifact(aux_start=80, aux_stop=90),
        offset_ticks=5.0,
        exposure_ticks=50.0,
    )
    outside_changed = _derive(
        _probe_artifact(aux_start=85, aux_stop=95),
        offset_ticks=5.0,
        exposure_ticks=50.0,
    )
    inside_changed = _derive(
        _probe_artifact(probe_stop=50),
        offset_ticks=5.0,
        exposure_ticks=50.0,
    )

    assert outside_changed == baseline
    assert inside_changed != baseline


def test_trigger_pulse_width_is_only_an_anchor_and_does_not_enter_context() -> None:
    short_trigger = _artifact(
        channels=("probe-lane", "trigger-lane"),
        ticks=(0, 10, 15, 20, 40, 100),
        masks=(0, 2, 0, 1, 0, 0),
        logical_outputs=(("probe", "probe-lane"), ("trigger", "trigger-lane")),
    )
    long_trigger = _artifact(
        channels=("probe-lane", "trigger-lane"),
        ticks=(0, 10, 20, 40, 70, 100),
        masks=(0, 2, 3, 2, 0, 0),
        logical_outputs=(("probe", "probe-lane"), ("trigger", "trigger-lane")),
    )

    assert _derive(short_trigger, offset_ticks=5, exposure_ticks=50) == _derive(
        long_trigger,
        offset_ticks=5,
        exposure_ticks=50,
    )


def test_target_abi_identity_is_part_of_an_otherwise_equal_context() -> None:
    first = _derive(
        _probe_artifact(target_abi_fingerprint="a" * 64),
        offset_ticks=5,
        exposure_ticks=50,
    )
    second = _derive(
        _probe_artifact(target_abi_fingerprint="b" * 64),
        offset_ticks=5,
        exposure_ticks=50,
    )

    assert first.digital == second.digital
    assert first.buses == second.buses
    assert first.target_abi_fingerprint == "a" * 64
    assert second.target_abi_fingerprint == "b" * 64
    assert first != second


def test_any_repeat_cell_with_a_different_context_is_rejected() -> None:
    artifact = _artifact(
        channels=("probe-lane", "trigger-lane"),
        ticks=(0, 10, 20, 40, 60, 70, 100),
        masks=(0, 3, 1, 0, 2, 0, 0),
        logical_outputs=(("probe", "probe-lane"), ("trigger", "trigger-lane")),
    )

    with pytest.raises(ValueError, match="varies across repeat/scan cells"):
        _derive(artifact, offset_ticks=0, exposure_ticks=20, repeats=2)


def test_any_scan_cell_with_a_different_decoded_bus_context_is_rejected() -> None:
    selected_edge = TargetBusSegment(
        0,
        "bias",
        25,
        25,
        0,
        0,
        "edge",
        1,
        1,
        (0,),
        (0,),
    )
    artifact = _artifact(
        channels=("trigger-lane",),
        ticks=(0, 10, 20, 100),
        masks=(0, 1, 0, 0),
        logical_outputs=(("trigger", "trigger-lane"),),
        bus_names=("bias",),
        bus_safe_values=(512,),
        bus_segments=(selected_edge,),
        slot_kinds=("dac",),
        scan_points=((600,), (700,)),
    )

    with pytest.raises(ValueError, match="varies across repeat/scan cells"):
        plan = _cell_plan(artifact)
        derive_readout_physical_context(
            PulseCaptureBinding(artifact, plan.trigger_channel, plan),
            readout_event_index=0,
            integration_start_offset_seconds=0.0,
            integration_seconds=50 / CLOCK_HZ,
        )


def test_decoded_dac_edge_is_captured_without_member_or_latch_lines() -> None:
    edge = TargetBusSegment(
        0,
        "bias",
        25,
        25,
        600,
        600,
        "edge",
        0,
        0,
        (),
        (),
    )
    artifact = _artifact(
        channels=("trigger-lane",),
        ticks=(0, 10, 20, 100),
        masks=(0, 1, 0, 0),
        logical_outputs=(("trigger", "trigger-lane"),),
        bus_names=("bias",),
        bus_safe_values=(512,),
        bus_segments=(edge,),
    )

    context = _derive(artifact, offset_ticks=0, exposure_ticks=50)
    assert context.digital == ()
    assert context.buses == (BusReadoutTrace("bias", 512, ((16, 600),)),)


def test_finite_done_safe_transition_uses_delay_and_registered_tick() -> None:
    edge = TargetBusSegment(
        0,
        "bias",
        25,
        25,
        600,
        600,
        "edge",
        0,
        0,
        (),
        (),
    )
    artifact = _artifact(
        channels=("trigger-lane",),
        ticks=(0, 10, 20, 100),
        masks=(0, 1, 0, 0),
        logical_outputs=(("trigger", "trigger-lane"),),
        bus_names=("bias",),
        bus_safe_values=(512,),
        bus_segments=(edge,),
        bus_delays=(TargetBusDelay(0, 7),),
    )

    # The current pulse owner exposes a bounded trigger-normalized window, not
    # a materialized tuple of every physical bus assignment.  Relative to the
    # trigger at tick 10, the edge is visible at 25 + register(1) + delay(7)
    # = 33 (relative 23), and finite DONE returns SAFE at 100 + 1 + 7 = 108
    # (relative 98).
    full_window = _derive(artifact, offset_ticks=0, exposure_ticks=100)
    assert full_window.buses == (
        BusReadoutTrace("bias", 512, ((23, 600), (98, 512))),
    )

    ending_at_safe = _derive(artifact, offset_ticks=0, exposure_ticks=98)
    assert ending_at_safe.buses == (
        BusReadoutTrace("bias", 512, ((23, 600),)),
    )
    starting_at_safe = _derive(artifact, offset_ticks=98, exposure_ticks=2)
    assert starting_at_safe.buses == (BusReadoutTrace("bias", 512),)


def test_artifacts_with_different_done_safe_timing_have_different_contexts() -> None:
    edge = TargetBusSegment(
        0,
        "bias",
        25,
        25,
        600,
        600,
        "edge",
        0,
        0,
        (),
        (),
    )

    def artifact(final_tick: int) -> CompiledPulseArtifact:
        return _artifact(
            channels=("trigger-lane",),
            ticks=(0, 10, 20, final_tick),
            masks=(0, 1, 0, 0),
            logical_outputs=(("trigger", "trigger-lane"),),
            bus_names=("bias",),
            bus_safe_values=(512,),
            bus_segments=(edge,),
        )

    early_done = _derive(artifact(100), offset_ticks=0, exposure_ticks=100)
    late_done = _derive(artifact(120), offset_ticks=0, exposure_ticks=100)

    assert early_done.buses == (
        BusReadoutTrace("bias", 512, ((16, 600), (91, 512))),
    )
    assert late_done.buses == (
        BusReadoutTrace("bias", 512, ((16, 600),)),
    )
    assert early_done != late_done


def test_live_state_dac_ramp_fails_closed() -> None:
    ramp = TargetBusSegment(
        0,
        "bias",
        25,
        30,
        0,
        600,
        "ramp",
        0,
        0,
        (),
        (),
    )
    artifact = _artifact(
        channels=("trigger-lane",),
        ticks=(0, 10, 20, 100),
        masks=(0, 1, 0, 0),
        logical_outputs=(("trigger", "trigger-lane"),),
        bus_names=("bias",),
        bus_safe_values=(512,),
        bus_segments=(ramp,),
    )

    with pytest.raises(ValueError, match="cannot represent a live-state DAC ramp exactly"):
        _derive(artifact, offset_ticks=0, exposure_ticks=50)


def test_missing_qualified_integration_offset_fails_closed() -> None:
    artifact = _probe_artifact()
    plan = _cell_plan(artifact)
    with pytest.raises(ValueError, match="qualified integration start offset"):
        derive_readout_physical_context(
            PulseCaptureBinding(artifact, plan.trigger_channel, plan),
            readout_event_index=0,
            integration_start_offset_seconds=None,
            integration_seconds=20 / CLOCK_HZ,
        )


def test_real_target_context_contains_only_logical_outputs_and_decoded_buses() -> None:
    document = load_pulse_document(IMAGING_TEMPLATE)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=CLOCK_HZ,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    plan = _cell_plan(artifact, readout_events=3)
    context = derive_readout_physical_context(
        PulseCaptureBinding(artifact, plan.trigger_channel, plan),
        readout_event_index=1,
        integration_start_offset_seconds=0.0,
        integration_seconds=1e-3,
    )

    assert tuple(item.output_key for item in context.digital) == tuple(
        key
        for key, _lane in artifact.target_ir.logical_digital_outputs
        if key != "ch11"
    )
    assert tuple(item.bus_name for item in context.buses) == tuple(
        sorted(artifact.target_ir.bus_names)
    )
    assert not any(item.output_key.startswith("da_") for item in context.digital)


def test_context_canonical_round_trip_preserves_the_value() -> None:
    context = _derive(_probe_artifact(), offset_ticks=9.5, exposure_ticks=20.5)
    restored = readout_physical_context_from_tree(
        readout_physical_context_to_tree(context)
    )

    assert restored == context
