"""Current formal MOT-field product flow over the virtual installation."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets

from gui_user_flow import (
    choose_combo_text,
    configure_offscreen_fast_path,
    current_logic_editor,
    replace_spin_value,
    require_offscreen_platform,
    until,
    visible_form_widgets,
    widget_gone,
)
from zlc_frontend.qt_widgets import ensure_qt_app
from zlc_plot import AxisRef, FacetGridPlot, ImagePlot, PlotKind
from zlc_neutral_atom.device_types import (
    CAPABILITY_MOT_FIELD_CAPTURE,
    CAPABILITY_PULSE_EXECUTE,
)
from zlc_neutral_atom.logic_nodes.mot_field.logic_node import LOGIC_NODE
from zlc_neutral_atom.logic_nodes.mot_field.mot_field import (
    MotFieldRequest,
    build_mot_scan_program,
)
from zlc_workbench.task_console.console_records import console_signal_key
from zlc_pulse import load_pulse_document


ROOT = Path(__file__).resolve().parents[1]


def test_mot_leaf_uses_one_stable_capability_bound_request() -> None:
    request = LOGIC_NODE.build_request(
        {
            "pulse": "mot_field_template.json",
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "span": 2.0,
            "points": 2,
            "roi_cx": None,
            "roi_cy": None,
            "roi_radius": 8.0,
            "camera_instance_id": "mot-camera",
            "sequencer_instance_id": "sequencer",
        }
    )
    assert isinstance(request, MotFieldRequest)
    assert request.camera_instance_id == "mot-camera"
    assert request.sequencer_instance_id == "sequencer"
    assert not hasattr(request, "camera_role")
    assert LOGIC_NODE.device_requirements == (
        ("camera_instance_id", (CAPABILITY_MOT_FIELD_CAPTURE,)),
        ("sequencer_instance_id", (CAPABILITY_PULSE_EXECUTE,)),
    )
    assert tuple(output.name for output in LOGIC_NODE.outputs) == (
        "grid",
        "mot_field",
        "scan",
    )
    assert LOGIC_NODE.ui_contributions == ()

    program = build_mot_scan_program(
        load_pulse_document(ROOT / "pulses" / request.pulse),
        center_x=request.center_x,
        center_y=request.center_y,
        center_z=request.center_z,
        span=request.span,
        points=request.points,
    )
    assert program.point_table.row_count == 8
    assert program.grid_topology.logical_shape == (2, 2, 2)


def _faceted_card(console, signal_key: str):
    for card in console.cards:
        if card.config.signal != signal_key or card.config.kind is not PlotKind.FACET_GRID:
            continue
        if card.host is not None and card.plot_widget is not None:
            return card
    return None


def _presented_value(card):
    return None if card is None else card.presented_value


def test_mot_field_form_runs_live_and_final_named_axis_grids(tmp_path: Path) -> None:
    """The visible Start path must draw data, not merely create a blank card."""

    configure_offscreen_fast_path()
    application = ensure_qt_app()
    require_offscreen_platform(application)
    from task_console import _StandaloneTaskConsoleFlow, _build_parser

    workspace = tmp_path / "workspace"
    pulses = workspace / "pulses"
    pulses.mkdir(parents=True)
    for name in ("imaging_template.json", "mot_field_template.json"):
        shutil.copy2(ROOT / "pulses" / name, pulses / name)
    flow = _StandaloneTaskConsoleFlow(
        _build_parser().parse_args(
            [
                "--workspace",
                str(workspace),
                "--name",
                "mot-field-current",
                "--seed",
                "37",
            ]
        )
    )
    devices = flow.open()
    console_wrapper = None
    try:
        QtTest.QTest.mouseClick(devices.apply_button, QtCore.Qt.LeftButton)
        until(
            application,
            lambda: flow.console is not None or flow.failure is not None,
            timeout=15.0,
        )
        assert flow.failure is None
        console = flow.console
        console_wrapper = console.window()
        add = next(
            button
            for button in console.findChildren(QtWidgets.QPushButton)
            if button.text() == "Add Panel"
        )

        choose_combo_text(
            console.kind_combo,
            "Task: Optimize MOT field",
            application,
        )
        QtTest.QTest.mouseClick(add, QtCore.Qt.LeftButton)
        row = console.logic_nodes[-1]
        editor = current_logic_editor(console, application)
        widgets = visible_form_widgets(editor)
        replace_spin_value(widgets["points"], "7")

        QtTest.QTest.mouseClick(
            editor.form.start_button,
            QtCore.Qt.LeftButton,
        )
        live_key = console_signal_key(row.node.node_id, "grid")
        final_key = console_signal_key(row.node.node_id, "mot_field")
        until(
            application,
            lambda: (
                (card := _faceted_card(console, live_key)) is not None
                and _presented_value(card) is not None
            ),
            timeout=20.0,
        )
        live_card = _faceted_card(console, live_key)
        live_value = live_card.presented_value
        first_live_publication = live_card.presented_publication
        assert live_value is not None and first_live_publication is not None
        first_live_config = live_card.current_plot_config()
        assert first_live_config is not None
        first_live_spec, _parameters = first_live_config
        assert isinstance(first_live_spec, FacetGridPlot)
        assert isinstance(first_live_spec.cell, ImagePlot)
        assert first_live_spec.facet == AxisRef.point_dimension("mot-field.da_z")
        assert first_live_spec.cell.x == AxisRef.point_dimension("mot-field.da_x")
        assert first_live_spec.cell.y == AxisRef.point_dimension("mot-field.da_y")
        first_live_revision = live_value.snapshot.ref.revision.value
        first_live_coverage = live_value.coverage
        assert first_live_coverage is not None
        assert 0 < first_live_coverage.written_cells < 343
        assert first_live_coverage.total_cells == 343
        assert live_value.transient
        live_schema = live_value.schema
        assert live_schema.grid_topology is not None
        assert live_schema.grid_topology.logical_shape == (7, 7, 7)
        assert live_schema.physical_shape == (1, 343, 1)
        assert tuple(
            (column.coordinate_id.value, column.role.value)
            for column in live_schema.point_table.columns
        ) == (
            ("mot-field.da_x", "scan-point"),
            ("mot-field.da_y", "scan-point"),
            ("mot-field.da_z", "scan-point"),
        )
        until(
            application,
            lambda: (
                (card := _faceted_card(console, live_key)) is not None
                and (value := _presented_value(card)) is not None
                and value.snapshot.ref.revision.value > first_live_revision
                and card.presented_publication.event_ref.sequence
                > first_live_publication.event_ref.sequence
            ),
            timeout=30.0,
        )
        second_live_card = _faceted_card(console, live_key)
        second_live_value = second_live_card.presented_value
        second_live_publication = second_live_card.presented_publication
        assert second_live_value is not None and second_live_publication is not None
        assert second_live_card.current_plot_config()[0] == first_live_spec
        second_live_coverage = second_live_value.coverage
        assert second_live_coverage is not None
        assert (
            second_live_publication.event_ref.stream_id
            == first_live_publication.event_ref.stream_id
        )
        assert (
            second_live_publication.event_ref.generation
            == first_live_publication.event_ref.generation
        )
        assert (
            second_live_publication.event_ref.sequence
            > first_live_publication.event_ref.sequence
        )
        assert second_live_value.snapshot.ref.revision.value > first_live_revision
        assert (
            second_live_coverage.written_cells
            > first_live_coverage.written_cells
        )
        assert second_live_coverage.total_cells == 343
        assert second_live_value.transient
        until(
            application,
            lambda: (
                row.status_label.text().startswith("done")
                # The generic form re-enables its Start action at terminal and
                # may therefore settle its local status to ``ready``.  The
                # row is the lifecycle owner; the editor only needs to be
                # non-running/non-error while the FINAL value is visible.
                and not editor.form.stop_button.isEnabled()
                and not editor.form.status.text().lower().startswith("error")
                and (card := _faceted_card(console, final_key)) is not None
                and _presented_value(card) is not None
            ),
            timeout=90.0,
        )
        final_card = _faceted_card(console, final_key)
        final_value = final_card.presented_value
        assert final_value is not None
        final_config = final_card.current_plot_config()
        assert final_config is not None and final_config[0] == first_live_spec
        final_schema = final_value.schema
        assert final_schema is live_schema
        assert final_schema.fingerprint == live_schema.fingerprint
        assert final_schema.grid_topology is not None
        assert final_schema.grid_topology.logical_shape == (7, 7, 7)
        assert final_schema.physical_shape == (1, 343, 1)
        assert final_value.coverage is None or final_value.coverage.complete
        assert tuple(
            column.coordinate_id.value
            for column in final_schema.point_table.columns
        ) == (
            "mot-field.da_x",
            "mot-field.da_y",
            "mot-field.da_z",
        )
        assert float(np.ptp(final_value.snapshot.block.values)) > 0.0
    finally:
        if not widget_gone(console_wrapper):
            console_wrapper.close()
            until(
                application,
                lambda: widget_gone(console_wrapper),
                timeout=15.0,
            )
        flow.close()
        application.processEvents(QtCore.QEventLoop.AllEvents, 20)
