"""Pulse preview projects once into the public zlc_plot vocabulary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zlc_plot import PulseTimelineData, PulseTimelinePlot, RasterPlotHost
from zlc_pulse import (
    PORT_DIGITAL,
    PulseExecutionForm,
    RepeatRegion,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_workbench.pulse_editor.preview_projection import (
    pulse_preview_status,
    pulse_timeline_plot,
    recommended_pulse_size,
)
from zlc_workbench.pulse_editor.session import project_pulse_preview


ROOT = Path(__file__).parents[1]


def test_projection_keeps_typed_row_order_and_show_off_is_only_a_view_filter():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    timeline = project_pulse_preview(document)

    active, active_spec = pulse_timeline_plot(
        timeline,
        include_off_rows=False,
    )
    all_rows, all_spec = pulse_timeline_plot(
        timeline,
        include_off_rows=True,
    )

    expected_active_digital = tuple(
        row.row_id
        for row in timeline.rows
        if row.port_kind == "digital" and row.active
    )
    expected_all_digital = tuple(
        row.row_id for row in timeline.rows if row.port_kind == "digital"
    )
    expected_active_dac = tuple(
        row.row_id
        for row in timeline.rows
        if row.port_kind == "dac" and row.active
    )
    assert isinstance(active, PulseTimelineData)
    assert isinstance(active_spec, PulseTimelinePlot)
    assert active_spec == all_spec
    assert tuple(item.channel_id for item in active.channels) == expected_active_digital
    assert tuple(item.channel_id for item in all_rows.channels) == expected_all_digital
    assert tuple(item.name for item in active.analog_traces) == expected_active_dac
    assert timeline.rows == project_pulse_preview(document).rows
    assert pulse_preview_status(timeline, include_off_rows=False).startswith(
        f"{len(expected_active_digital)}/{len(expected_all_digital)} plotted"
    )


def test_all_off_projection_retains_first_digital_reference_and_active_dac_rows():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    digital_lanes = {
        port.lanes[0] for port in document.target.ports if port.kind == PORT_DIGITAL
    }
    all_off = replace(
        document,
        periods=tuple(
            replace(
                period,
                states=tuple(
                    0 if lane in digital_lanes else value
                    for lane, value in zip(
                        document.target.raw_lanes,
                        period.states,
                        strict=True,
                    )
                ),
            )
            for period in document.periods
        ),
    )
    timeline = project_pulse_preview(all_off)
    projected, _spec = pulse_timeline_plot(timeline, include_off_rows=False)
    digital = tuple(row for row in timeline.rows if row.port_kind == "digital")
    active_dac = tuple(
        row for row in timeline.rows if row.port_kind == "dac" and row.active
    )

    assert tuple(item.channel_id for item in projected.channels) == (
        digital[0].row_id,
    )
    assert tuple(trace.name for trace in projected.analog_traces) == tuple(
        row.row_id for row in active_dac
    )


def test_repeat_projection_preserves_the_formal_three_state_semantics():
    base = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    first = base.periods[0].period_id
    second = base.periods[1].period_id
    last = base.periods[-1].period_id

    cases = (
        (replace(base, repeat=None), "repeat ∞", ("×∞",)),
        (
            replace(base, repeat=RepeatRegion(second, last, 3)),
            "repeat ∞ + P2-P6 x3",
            ("×∞", "×3"),
        ),
        (
            replace(base, repeat=RepeatRegion(first, last, 4)),
            "repeat P1-P6 x4",
            ("×4",),
        ),
    )
    for document, expected_summary, expected_labels in cases:
        timeline = project_pulse_preview(document)
        projected, _spec = pulse_timeline_plot(
            timeline,
            include_off_rows=False,
        )
        assert projected.repeat_notation == expected_summary
        assert tuple(marker.label for marker in projected.repeat_markers) == (
            expected_labels
        )
        assert pulse_preview_status(
            timeline,
            include_off_rows=False,
        ).endswith(f"| {expected_summary}")

    no_repeat = project_pulse_preview(replace(base, repeat=None))
    projected, _spec = pulse_timeline_plot(no_repeat, include_off_rows=False)
    assert projected.repeat_markers[0].start == 0.0
    assert projected.repeat_markers[0].stop == pytest.approx(
        no_repeat.duration_ticks / no_repeat.clock_hz
    )


def test_projection_preserves_scan_overlay_and_compiler_owned_source_proof():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    if document.repeat is None and len(document.periods) >= 2:
        document = replace(
            document,
            repeat=RepeatRegion(
                document.periods[0].period_id,
                document.periods[1].period_id,
                3,
            ),
        )
    timeline = project_pulse_preview(document)
    projected, _spec = pulse_timeline_plot(timeline, include_off_rows=True)

    assert len(projected.scan_dac_segments) == len(document.scan_parameters)
    assert tuple(item.number for item in projected.scan_dac_segments) == tuple(
        range(1, len(document.scan_parameters) + 1)
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=1e9 / document.time_step_ns,
        execution_form=PulseExecutionForm.STATIC_REFERENCE_POINT,
        live_target=document.target,
    )
    assert artifact.source_document_digest == timeline.source_document_digest


def test_zlc_plot_host_owns_fixed_geometry_viewport_and_export_surface():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    timeline = project_pulse_preview(document)
    data, spec = pulse_timeline_plot(timeline, include_off_rows=True)
    size = recommended_pulse_size(timeline, include_off_rows=True)
    assert size in {"1x2", "2x2", "4x2", "1x4", "2x4", "4x4", "4x8", "8x4", "8x8"}

    host = RasterPlotHost.from_plot(data, spec, size="1x2")
    try:
        home = host.wait_for_front(timeout=10.0)
        home_bounds = home.interaction.axes[0].bounds
        low, high = home.interaction.axes[0].x_limits
        span = high - low
        changed = host.set_x_limits(
            low + 0.2 * span,
            high - 0.2 * span,
        ).result(timeout=10.0).front
        assert changed.interaction.axes[0].bounds == pytest.approx(
            home_bounds,
            abs=1e-15,
        )
        assert changed.logical_size == home.logical_size
        assert changed.identity.kind == "pulse_timeline"
        assert changed.identity.preset == "1x2"
    finally:
        assert host.close(timeout=10.0)
