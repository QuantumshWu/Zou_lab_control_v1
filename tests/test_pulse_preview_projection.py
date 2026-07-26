"""Current PulseTimelineDocument is the sole source of preview render facts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zlc_pulse import (
    PORT_DIGITAL,
    PulseExecutionForm,
    RepeatRegion,
    compile_pulse_artifact,
    load_pulse_document,
)
from zlc_frontend.matplotlib_render import (
    render_pulse_timeline_panel,
)
from zlc_frontend.render import DocumentInputIdentity
from zlc_frontend.render_style import (
    panel_axes_bounds,
    panel_display_size,
)
from zlc_workbench.pulse_editor.session import project_pulse_preview
from zlc_workbench.pulse_editor.preview_projection import (
    pulse_preview_status,
    pulse_timeline_render_kwargs,
)


ROOT = Path(__file__).parents[1]


def test_projection_keeps_typed_row_order_and_show_off_is_only_a_view_filter():
    document = load_pulse_document(
        ROOT / "pulses" / "imaging_template.json"
    )
    timeline = project_pulse_preview(document)

    active = pulse_timeline_render_kwargs(
        timeline,
        include_off_rows=False,
    )
    all_rows = pulse_timeline_render_kwargs(
        timeline,
        include_off_rows=True,
    )

    expected_active_digital = [
        row.row_id
        for row in timeline.rows
        if row.port_kind == "digital" and row.active
    ]
    expected_all_digital = [
        row.row_id for row in timeline.rows if row.port_kind == "digital"
    ]
    expected_active_dac = [
        row.row_id
        for row in timeline.rows
        if row.port_kind == "dac" and row.active
    ]
    assert active["channels"] == expected_active_digital
    assert all_rows["channels"] == expected_all_digital
    assert [item["name"] for item in active["analog_traces"]] == expected_active_dac
    assert timeline.rows == project_pulse_preview(document).rows
    assert pulse_preview_status(
        timeline, include_off_rows=False
    ).startswith(f"{len(expected_active_digital)}/{len(expected_all_digital)} plotted")


def test_all_off_projection_retains_first_digital_reference_and_active_dac_rows():
    document = load_pulse_document(
        ROOT / "pulses" / "imaging_template.json"
    )
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
    projected = pulse_timeline_render_kwargs(timeline, include_off_rows=False)
    digital = tuple(row for row in timeline.rows if row.port_kind == "digital")
    active_dac = tuple(
        row for row in timeline.rows if row.port_kind == "dac" and row.active
    )

    assert projected["channels"] == [digital[0].row_id]
    assert [trace["name"] for trace in projected["analog_traces"]] == [
        row.row_id for row in active_dac
    ]
    assert pulse_preview_status(timeline, include_off_rows=False).startswith(
        f"1/{len(digital)} plotted (active channels)"
    )


def test_repeat_projection_preserves_the_formal_three_state_semantics():
    base = load_pulse_document(
        ROOT / "pulses" / "imaging_template.json"
    )
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
        projected = pulse_timeline_render_kwargs(
            timeline,
            include_off_rows=False,
        )
        assert projected["repeat_notation"] == expected_summary
        assert tuple(marker[2] for marker in projected["repeat_markers"]) == (
            expected_labels
        )
        assert pulse_preview_status(
            timeline,
            include_off_rows=False,
        ).endswith(f"| {expected_summary}")

    no_bracket = pulse_timeline_render_kwargs(
        project_pulse_preview(replace(base, repeat=None)),
        include_off_rows=False,
    )["repeat_markers"]
    assert no_bracket[0][0] == 0.0
    assert no_bracket[0][1] == pytest.approx(
        project_pulse_preview(replace(base, repeat=None)).duration_ticks
        / project_pulse_preview(replace(base, repeat=None)).clock_hz
    )


def test_projection_preserves_typed_scan_overlay_and_authored_repeat_geometry():
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

    projected = pulse_timeline_render_kwargs(
        timeline,
        include_off_rows=True,
        size="2x2",
    )

    assert projected["size"] == "2x2"
    assert len(projected["scan_dac_segments"]) == len(document.scan_parameters)
    assert [item["number"] for item in projected["scan_dac_segments"]] == list(
        range(1, len(document.scan_parameters) + 1)
    )
    if document.repeat is not None:
        assert projected["repeat_markers"]
        assert projected["repeat_notation"] == "repeat ∞ + P1-P2 x3"

    # The projection does not compile or mutate; its target proof remains the
    # exact current compiler result owned by the timeline.
    artifact = compile_pulse_artifact(
        document,
        clock_hz=1e9 / document.time_step_ns,
        execution_form=PulseExecutionForm.STATIC_REFERENCE_POINT,
        live_target=document.target,
    )
    assert artifact.source_document_digest == timeline.source_document_digest


def test_pulse_viewport_never_participates_in_formal_plot_layout():
    """Ticks and off-screen scan badges cannot move the plot chrome."""

    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    timeline = project_pulse_preview(document)
    projected = pulse_timeline_render_kwargs(
        timeline,
        include_off_rows=True,
        size="1x2",
    )
    source = DocumentInputIdentity(
        "pulse-layout-contract",
        0,
        timeline.fingerprint,
    )
    raster, home = render_pulse_timeline_panel(
        **projected,
        document_input=source,
    )
    left, bottom, width, height = panel_axes_bounds("1x2", kind="pulse")
    expected_plot_bounds = (
        left,
        1.0 - (bottom + height),
        left + width,
        1.0 - bottom,
    )
    assert home.viewport.plot_bounds == pytest.approx(
        expected_plot_bounds,
        abs=1e-15,
    )
    assert (raster.width, raster.height) == panel_display_size(
        "1x2",
        kind="pulse",
    )
    # Exact formal main geometry: 122 px pulse left margin, 480 px data box,
    # 96 px right margin, displayed at the single 0.7 panel scale.
    assert panel_display_size("1x2", kind="pulse") == (489, 231)

    home_left, home_right = home.viewport.home_x_limits
    span = home_right - home_left
    viewports = (
        (-10.0 * span, -9.98 * span),  # all tick labels blank/negative
        (0.5, 0.5 + 0.02 * span),      # scan badges well outside the view
        (1_000.0, 1_000.0 + 0.02 * span),
        (-span, 0.5 * span),
        (0.0, max(span * 0.001, 1e-12)),
    )
    for revision, limits in enumerate(viewports, start=1):
        candidate_raster, candidate = render_pulse_timeline_panel(
            **projected,
            document_input=source,
            display_revision=revision,
            x_limits=limits,
        )
        assert candidate.viewport.plot_bounds == pytest.approx(
            expected_plot_bounds,
            abs=1e-15,
        )
        assert (candidate_raster.width, candidate_raster.height) == (
            raster.width,
            raster.height,
        )
