from __future__ import annotations

from pathlib import Path

import pytest

from zlc_pulse import (
    PORT_DIGITAL,
    PulseExecutionForm,
    build_pulse_playback,
    compile_pulse_artifact,
    load_pulse_document,
)


ROOT = Path(__file__).parents[1]


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
        ("pulse_test.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE),
    ),
)
def test_playback_rising_edges_are_the_compiled_trigger_schedule(filename, form):
    document = load_pulse_document(ROOT / "pulses" / filename)
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
        edge.tick_from_run_start for edge in artifact.trigger_schedules[0].edges
    )
    assert playback.logical_duration == pytest.approx(
        artifact.target_ir.duration_seconds
    )


def test_continuous_playback_is_cyclic_but_never_fabricates_a_finite_schedule():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.CONTINUOUS_MONITOR,
    )

    playback = build_pulse_playback(artifact, name=document.name)

    assert playback.repeat_forever
    assert artifact.trigger_schedules == ()
    assert playback.base_pulses()
