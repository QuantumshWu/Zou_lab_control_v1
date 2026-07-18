from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zlc_pulse import (
    CompiledPulseArtifact,
    MAX_MATERIALIZED_PLAYBACK_PULSES,
    OutputDelay,
    PORT_DIGITAL,
    PulseExecutionForm,
    RepeatRegion,
    TargetIR,
    build_pulse_playback,
    compile_pulse_artifact,
    load_pulse_document,
    pack_target_ir,
)


ROOT = Path(__file__).parents[1]


def _document_path(filename: str) -> Path:
    if filename == "imaging_template.json":
        return ROOT / "zlc_neutral_atom" / "assets" / filename
    return ROOT / "pulses" / filename


def _active_digital_lane(document):
    return next(
        port.lanes[0]
        for port in document.target.ports
        if port.kind == PORT_DIGITAL
        and any(
            period.states[document.target.raw_lanes.index(port.lanes[0])]
            for period in document.periods
        )
    )


@pytest.mark.parametrize(
    ("filename", "form"),
    (
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE),
    ),
)
def test_playback_rising_edges_are_the_compiled_trigger_schedule(filename, form):
    document = load_pulse_document(_document_path(filename))
    trigger = _active_digital_lane(document)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=form,
        trigger_channels=(trigger,),
    )

    playback = build_pulse_playback(artifact, name=document.name)
    starts = tuple(
        round(pulse.start * artifact.target_ir.clock_hz)
        for pulse in playback.base_pulses()
        if pulse.channel == trigger and pulse.value
    )

    assert starts == tuple(
        edge.tick_from_run_start
        for edge in artifact.trigger_schedules[0].iter_edges()
    )
    assert playback.logical_duration == pytest.approx(
        artifact.target_ir.duration_seconds
    )


def test_continuous_playback_is_cyclic_but_never_fabricates_a_finite_schedule():
    document = load_pulse_document(_document_path("imaging_template.json"))
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.CONTINUOUS_MONITOR,
    )

    playback = build_pulse_playback(artifact, name=document.name)

    assert playback.repeat_forever
    assert artifact.trigger_schedules == ()
    assert playback.base_pulses()


def _repeated_imaging_document(count=12):
    document = load_pulse_document(_document_path("imaging_template.json"))
    return replace(
        document,
        repeat=RepeatRegion(
            document.periods[0].period_id,
            document.periods[-1].period_id,
            count,
        ),
    )


def test_full_document_repeat_retains_point_loop_trigger_groups():
    document = _repeated_imaging_document()
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    playback = build_pulse_playback(artifact)
    schedule = artifact.trigger_schedules[0]

    assert schedule.loop_count == 12
    assert schedule.full_point_loop
    assert tuple(edge.loop_iteration for edge in schedule.iter_edges()) == tuple(
        repeat for repeat in range(12) for _event in range(3)
    )
    assert playback.trigger_group_sizes(("ch11",)) == (3,) * 12
    assert tuple(
        round(pulse.start * artifact.target_ir.clock_hz)
        for pulse in playback.effective_pulses()
        if pulse.channel == "ch11" and pulse.value
    ) == tuple(edge.tick_from_run_start for edge in schedule.iter_edges())


def test_delay_unroll_keeps_source_groups_while_ticks_remain_physical():
    document = _repeated_imaging_document()
    delayed = replace(
        document,
        delays=(OutputDelay("ch11", 100, "ns"),),
    )
    artifact = compile_pulse_artifact(
        delayed,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    schedule = artifact.trigger_schedules[0]
    playback = build_pulse_playback(artifact)

    assert artifact.target_ir.loop_count == 1  # wire lowering was unrolled
    assert schedule.loop_count == 12           # source execution groups survive
    assert schedule.full_point_loop
    assert playback.trigger_group_sizes(("ch11",)) == (3,) * 12
    assert tuple(
        round(pulse.start * artifact.target_ir.clock_hz)
        for pulse in playback.effective_pulses()
        if pulse.channel == "ch11" and pulse.value
    ) == tuple(edge.tick_from_run_start for edge in schedule.iter_edges())


def test_partial_repeat_requires_explicit_virtual_shot_semantics():
    document = load_pulse_document(_document_path("imaging_template.json"))
    partial = replace(
        document,
        repeat=RepeatRegion(
            document.periods[1].period_id,
            document.periods[-2].period_id,
            2,
        ),
    )
    artifact = compile_pulse_artifact(
        partial,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
    )
    playback = build_pulse_playback(artifact)

    assert not artifact.trigger_schedules[0].full_point_loop
    with pytest.raises(ValueError, match="full-point source loop"):
        playback.trigger_group_sizes(("ch11",))


def test_virtual_shot_grouping_rejects_multiple_trigger_lines():
    """Per-line delays can interleave loop groups, so counts alone are unsafe."""

    document = _repeated_imaging_document(count=2)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11", "ch10"),
    )
    playback = build_pulse_playback(artifact)

    with pytest.raises(ValueError, match="exactly one trigger channel"):
        playback.trigger_group_sizes(("ch11", "ch10"))


def test_playback_rejects_an_unbounded_materialized_compact_projection():
    loops = MAX_MATERIALIZED_PLAYBACK_PULSES + 1
    ir = TargetIR(
        clock_hz=50e6,
        target_abi_fingerprint="a" * 64,
        channels=("trigger",),
        ticks=(0, 5, 10),
        masks=(1, 0, 0),
        duration_seconds=10 * loops / 50e6,
        repeat_forever=False,
        loop_start_index=0,
        loop_end_tick=10,
        loop_count=loops,
        tick_slot_coeffs=((), (), ()),
        channel_delays=(0,),
        logical_digital_outputs=(("trigger", "trigger"),),
    )
    artifact = CompiledPulseArtifact(
        "b" * 64,
        "test-compiler",
        PulseExecutionForm.STATIC_ONCE,
        ir,
        pack_target_ir(ir),
        (),
    )
    with pytest.raises(ValueError, match="physical digital playback requires"):
        build_pulse_playback(artifact)
