"""Current formal PulseGUI scan workspace: strict candidates and worker intents."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pytest

from zlc_pulse import FIELD_DAC, FIELD_DURATION, PORT_DAC, PulseFieldRef
from zlc_pulse.authoring import cycle_field_binding, new_period, insert_period, set_analog_action
from zlc_pulse.document import AnalogStep
from zlc_pulse import load_deployed_pulse_target, new_pulse_document
from zlc_workbench.pulse import PulseEditorSession
from zlc_workbench.pulse_editor.controller import PulseEditorController
from zlc_workbench.pulse_editor.scan_workspace import (
    default_scan_program,
    execute_scan_program,
    load_scan_array,
    save_scan_array,
    scan_column_specs,
)


def _pump_until(controller, predicate, *, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    snapshot = controller.pump()
    while not predicate(snapshot) and time.monotonic() < deadline:
        time.sleep(0.005)
        snapshot = controller.pump()
    assert predicate(snapshot), snapshot
    return snapshot


def _controller_with_duration_columns(count: int = 2) -> PulseEditorController:
    target = load_deployed_pulse_target()
    controller = PulseEditorController(
        PulseEditorSession.new(target, time_step_ns=20)
    )
    first = controller.snapshot().document.periods[0].period_id
    controller.cycle_binding(PulseFieldRef(FIELD_DURATION, first))
    for _ in range(1, count):
        period_id = controller.add_period(duration=1, unit="us").period_ids[0]
        controller.cycle_binding(PulseFieldRef(FIELD_DURATION, period_id))
    return controller


def _close(controller: PulseEditorController) -> None:
    controller.request_close()
    _pump_until(controller, lambda value: value.close_complete)
    assert controller.worker_idle


def test_scan_program_requires_exact_finite_two_dimensional_parameter_matrix():
    controller = _controller_with_duration_columns(2)
    try:
        assert not controller.snapshot().scan_workspace.source_dirty
        document = controller.snapshot().document
        result = execute_scan_program(
            document,
            "scan_table = np.array([[20, 40], [60, 80]])\n"
            "assert n_slots == 2\n",
        )
        assert result.candidate.table.columns == tuple(
            item.parameter_id for item in document.scan_parameters
        )
        assert result.candidate.table.rows == ((20, 40), (60, 80))

        with pytest.raises(ValueError, match="exactly two-dimensional"):
            execute_scan_program(document, "scan_table = [1, 2]\n")
        with pytest.raises(ValueError, match="require 2"):
            execute_scan_program(document, "scan_table = [[1], [2]]\n")
        with pytest.raises(ValueError, match="finite"):
            execute_scan_program(document, "scan_table = [[20, float('nan')]]\n")
        with pytest.raises(TypeError, match="real numeric"):
            execute_scan_program(document, "scan_table = [[True, False]]\n")
    finally:
        _close(controller)


def test_template_defaults_come_from_nominal_clock_units_and_real_dac_range():
    target = load_deployed_pulse_target()
    document = new_pulse_document(target, time_step_ns=20)
    first = document.periods[0]
    second = new_period(document, duration=2, unit="us")
    document = insert_period(document, period=second).document
    document = cycle_field_binding(
        document, PulseFieldRef(FIELD_DURATION, second.period_id), cascade=True
    ).document
    dac = next(port for port in target.ports if port.kind == PORT_DAC)
    document = set_analog_action(
        document,
        first.period_id,
        dac.key,
        AnalogStep(dac.key, "edge", 0),
        cascade=True,
    ).document
    document = cycle_field_binding(
        document,
        PulseFieldRef(FIELD_DAC, first.period_id, dac.key),
        cascade=True,
    ).document

    specs = scan_column_specs(document)
    duration = next(spec for spec in specs if not spec.is_dac)
    analog = next(spec for spec in specs if spec.is_dac)
    assert duration.unit == "ns"
    assert duration.lo == pytest.approx(20.0)
    assert duration.hi >= 4000.0
    assert (analog.lo, analog.hi) == tuple(float(v) for v in dac.signed_range)
    source = default_scan_program(document)
    assert duration.name == "duration_p2"
    assert duration.name in source
    assert analog.name.endswith("_p1")
    assert analog.name in source
    assert "Period 1" in source and "Period 2" in source
    assert all(
        item.parameter_id == spec.name
        for item, spec in zip(document.scan_parameters, specs, strict=True)
    )


def test_controller_worker_commits_generated_recipe_in_stable_parameter_order():
    controller = _controller_with_duration_columns(2)
    try:
        document = controller.snapshot().document
        columns = tuple(item.parameter_id for item in document.scan_parameters)
        display_names = tuple(spec.name for spec in scan_column_specs(document))
        source = "import numpy as np\nscan_table = np.array([[20, 40], [60, 80]])\n"
        controller.generate_scan_source(source)
        snapshot = _pump_until(
            controller,
            lambda value: value.scan_workspace.busy_operation is None
            and value.document.scan_table is not None,
        )
        assert snapshot.document.scan_table.columns == columns
        assert snapshot.document.scan_recipe is not None
        assert snapshot.document.scan_recipe.source == source
        assert snapshot.scan_workspace.selected_source == "generated"
        assert snapshot.scan_workspace.selected_compatible
        assert not snapshot.scan_workspace.source_dirty
        assert snapshot.scan_workspace.table_text.splitlines()[0] == "   ".join(
            display_names
        )
        assert not any(
            parameter_id in snapshot.scan_workspace.table_text
            for parameter_id in columns
            if parameter_id not in display_names
        )

        controller.generate_scan_source("scan_table = [1, 2]\n")
        failed = _pump_until(
            controller,
            lambda value: value.scan_workspace.busy_operation is None
            and "Scan code error" in value.scan_workspace.diagnostic,
        )
        assert failed.document.scan_table == snapshot.document.scan_table
        assert failed.scan_workspace.source_dirty
    finally:
        _close(controller)


def test_loaded_and_generated_candidates_never_reconcile_after_slot_change(tmp_path: Path):
    controller = _controller_with_duration_columns(2)
    try:
        source = "scan_table = [[20, 40], [60, 80]]\n"
        controller.generate_scan_source(source)
        _pump_until(
            controller,
            lambda value: value.scan_workspace.busy_operation is None
            and value.document.scan_table is not None,
        )

        path = tmp_path / "loaded.csv"
        path.write_text("100,120\n140,160\n", encoding="utf-8")
        controller.load_scan_array(path)
        loaded = _pump_until(
            controller,
            lambda value: value.scan_workspace.busy_operation is None
            and value.scan_workspace.selected_source == "loaded",
        )
        assert loaded.scan_workspace.loaded is not None
        assert loaded.document.scan_table.rows == ((100, 120), (140, 160))

        removed = loaded.document.scan_parameters[-1].field
        controller.cycle_binding(removed)  # scan -> API changes the ordered schema
        stale = controller.snapshot()
        assert stale.document.scan_table is None
        assert stale.scan_workspace.selected_source == "loaded"
        assert stale.scan_workspace.loaded is not None
        assert not stale.scan_workspace.loaded.compatible
        assert stale.scan_workspace.generated is not None
        assert not stale.scan_workspace.generated.compatible
        assert stale.scan_workspace.table_text.startswith("STALE")
        assert stale.scan_workspace.loaded.table.columns == tuple(
            item.parameter_id for item in loaded.document.scan_parameters
        )

        with pytest.raises(ValueError, match="stale"):
            controller.save_scan_array(tmp_path / "must_not_save.npy")
        controller.select_scan_source("generated")
        assert controller.snapshot().document.scan_table is None
    finally:
        _close(controller)


def test_array_io_preserves_rows_without_one_dimensional_shape_guessing(tmp_path: Path):
    controller = _controller_with_duration_columns(1)
    try:
        document = controller.snapshot().document
        csv_path = tmp_path / "one_column.csv"
        csv_path.write_text("20\n40\n60\n", encoding="utf-8")
        loaded = load_scan_array(document, csv_path)
        assert loaded.candidate.table.rows == ((20,), (40,), (60,))

        text_path = tmp_path / "one_column.txt"
        text_path.write_text("80\n100\n", encoding="utf-8")
        assert load_scan_array(document, text_path).candidate.table.rows == ((80,), (100,))

        ambiguous = tmp_path / "one_dimensional.npy"
        np.save(ambiguous, np.asarray([20.0, 40.0, 60.0]))
        with pytest.raises(ValueError, match="exactly two-dimensional"):
            load_scan_array(document, ambiguous)

        saved = save_scan_array(loaded.candidate.table, tmp_path / "frozen")
        assert saved.suffix == ".npy"
        assert np.load(saved).shape == (3, 1)
        with pytest.raises(ValueError, match=".npy or .csv"):
            save_scan_array(loaded.candidate.table, tmp_path / "changed.txt")
    finally:
        _close(controller)


def test_program_and_array_file_operations_are_worker_observable(tmp_path: Path):
    controller = _controller_with_duration_columns(1)
    try:
        # Load Program treats .txt as trusted Python; the Edit-panel Load Array
        # treats the same suffix as a numeric row-delimited matrix.
        program = tmp_path / "scan.txt"
        program.write_text("scan_table = [[20], [40]]\n", encoding="utf-8")
        controller.load_scan_program(program)
        loaded_program = _pump_until(
            controller,
            lambda value: value.scan_workspace.busy_operation is None
            and value.scan_workspace.source_text.startswith("scan_table"),
        )
        assert loaded_program.scan_workspace.source_dirty

        controller.generate_scan_source()
        generated = _pump_until(
            controller,
            lambda value: value.scan_workspace.busy_operation is None
            and value.document.scan_table is not None,
        )
        assert not generated.scan_workspace.source_dirty

        controller.save_scan_array(tmp_path / "selected.csv")
        saved = _pump_until(
            controller,
            lambda value: value.scan_workspace.busy_operation is None
            and value.scan_workspace.diagnostic.startswith("Saved scan array"),
        )
        assert (tmp_path / "selected.csv").exists()
        assert saved.scan_workspace.selected_table == generated.document.scan_table

        current_document = tmp_path / "current.json"
        generated.document.save(current_document)
        imported = load_scan_array(generated.document, current_document)
        assert imported.candidate.table == generated.document.scan_table
        controller.load_scan_program(current_document)
        imported_by_formal_button = _pump_until(
            controller,
            lambda value: value.scan_workspace.busy_operation is None
            and value.scan_workspace.selected_source == "loaded",
        )
        assert imported_by_formal_button.scan_workspace.loaded_path == current_document

        bad = tmp_path / "compiled_program.json"
        bad.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="current PulseDocument"):
            load_scan_array(generated.document, bad)
    finally:
        _close(controller)
