"""Current formal MOT-field product flow over the virtual installation."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets

from zlc_data import AxisSourceRef
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
from zlc_frontend import AxisViewRole, PlotKind, ViewIntent
from zlc_frontend.figure import grid_facet_source
from zlc_frontend.qt_widgets import FigureSurfaceHost, ensure_qt_app
from zlc_workbench.task_console.console_records import console_signal_key


ROOT = Path(__file__).resolve().parents[1]


def _faceted_card(console, signal_key: str):
    for card in console.cards:
        if card.config.signal != signal_key or card.config.kind != PlotKind.GRID:
            continue
        if isinstance(card.board, FigureSurfaceHost) and card.board.faceted:
            return card
    return None


def _presented_value(card):
    if card is None:
        return None
    try:
        return card.frozen_render_value()
    except RuntimeError:
        return None


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
        QtTest.QTest.mouseClick(devices.lifecycle_button, QtCore.Qt.LeftButton)
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
                and card.board.showing_overview
                and card.board.overview_artifact is not None
            ),
            timeout=20.0,
        )
        live_card = _faceted_card(console, live_key)
        live_value = live_card.frozen_render_value()
        first_live_publication = live_card.frozen_render_publication()
        first_live_intent = live_card._presentation_provider(
            live_value,
            first_live_publication,
        )
        assert first_live_intent is None
        first_live_contract = live_card._presented_contract
        assert first_live_contract is not None
        first_live_view = first_live_contract.figure.view
        assert first_live_contract.figure.kind is PlotKind.GRID
        assert first_live_view is not None
        assert first_live_view.intent is ViewIntent.IMAGE
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
        grid_sources = tuple(
            AxisSourceRef.grid_dimension(column.coordinate_id)
            for column in live_schema.point_table.columns
        )
        assert first_live_view.binding(grid_sources[0]).role is AxisViewRole.IMAGE_X
        assert first_live_view.binding(grid_sources[1]).role is AxisViewRole.IMAGE_Y
        assert grid_facet_source(first_live_view) == grid_sources[2]
        until(
            application,
            lambda: (
                (card := _faceted_card(console, live_key)) is not None
                and (value := _presented_value(card)) is not None
                and value.snapshot.ref.revision.value > first_live_revision
                and card.frozen_render_publication().event_ref.sequence
                > first_live_publication.event_ref.sequence
            ),
            timeout=30.0,
        )
        second_live_card = _faceted_card(console, live_key)
        second_live_value = second_live_card.frozen_render_value()
        second_live_publication = second_live_card.frozen_render_publication()
        second_live_intent = second_live_card._presentation_provider(
            second_live_value,
            second_live_publication,
        )
        assert second_live_intent is None
        second_live_contract = second_live_card._presented_contract
        assert second_live_contract is not None
        assert second_live_contract.figure.view == first_live_view
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
                and editor.form.status.text().startswith("done")
                and (card := _faceted_card(console, final_key)) is not None
                and _presented_value(card) is not None
                and card.board.showing_overview
                and card.board.overview_artifact is not None
            ),
            timeout=90.0,
        )
        final_card = _faceted_card(console, final_key)
        final_value = final_card.frozen_render_value()
        final_overview = final_card.board.overview_artifact
        final_schema = final_value.schema
        assert final_schema is live_schema
        assert final_schema.fingerprint == live_schema.fingerprint
        assert final_schema.grid_topology is not None
        assert final_schema.grid_topology.logical_shape == (7, 7, 7)
        assert len(final_overview.regions) == len(
            final_schema.grid_topology.coordinate_domains[2]
        )
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
