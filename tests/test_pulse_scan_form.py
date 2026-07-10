"""PulseScan form exposes scan-slot and API-slot execution without schema fallbacks."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def _app(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _api_rows(target="1", current=5.0):
    return [("a1", "image_duration", "duration", target, "us", current)]


def _scan_rows():
    return [
        ("da_x", "dac", "da_x@0", "value", "da_x"),
        ("da_y", "dac", "da_y@0", "value", "da_y"),
    ]


def _widget(program_id="mot-v1", code="scan_table = np.column_stack((da_x, da_y))"):
    from Zou_lab_control.frontend.task_console import _PulseSlotsWidget

    widget = _PulseSlotsWidget()
    widget.rebuild(
        api_rows=_api_rows(),
        scan_rows=_scan_rows(),
        hardware_program=code,
        program_id=program_id,
    )
    return widget


def _labels(widget):
    from PyQt5 import QtWidgets
    return "\n".join(label.text() for label in widget.findChildren(QtWidgets.QLabel))


def test_selected_template_hardware_program_is_loaded_verbatim(_app):
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_SCAN_SLOT

    code = "da_x = np.arange(-4, 5)\nda_y = da_x[::-1]\nscan_table = np.column_stack((da_x, da_y))"
    widget = _widget(code=code)
    assert widget._sweep_kind == SWEEP_SCAN_SLOT
    assert widget._program_code.toPlainText() == code
    assert "da_x" in _labels(widget) and "da_y" in _labels(widget)


def test_form_value_has_one_explicit_strategy_and_one_program(_app):
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_SCAN_SLOT

    value = _widget().values_dict()
    assert set(value) == {"program_id", "api", "sweep_kind", "program"}
    assert value["program_id"] == "mot-v1"
    assert value["api"]["a1"] == pytest.approx(5.0)
    assert value["sweep_kind"] == SWEEP_SCAN_SLOT
    assert "da_x" in value["program"]


def test_api_only_template_is_a_valid_api_slot_sweep(_app):
    from Zou_lab_control.frontend.task_console import _PulseSlotsWidget
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_API_SLOT

    widget = _PulseSlotsWidget()
    widget.rebuild(api_rows=_api_rows(), scan_rows=[], program_id="probe-v1")
    value = widget.values_dict()
    assert value["sweep_kind"] == SWEEP_API_SLOT
    assert "image_duration" in value["program"]
    assert value["program"].strip()


def test_template_without_either_slot_cannot_start_a_scan(_app):
    from Zou_lab_control.frontend.task_console import _PulseSlotsWidget

    widget = _PulseSlotsWidget()
    widget.rebuild(api_rows=[], scan_rows=[], program_id="empty-v1")
    value = widget.values_dict()
    assert value["sweep_kind"] == "" and value["program"] == ""
    assert "bind at least one" in _labels(widget).lower()


def test_switching_strategy_preserves_each_program_buffer(_app):
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_API_SLOT, SWEEP_SCAN_SLOT

    widget = _widget(code="scan_table = [[1, 2], [3, 4]]")
    api_index = widget._sweep_combo.findData(SWEEP_API_SLOT)
    widget._sweep_combo.setCurrentIndex(api_index)
    assert widget._sweep_kind == SWEEP_API_SLOT
    widget._program_code.setPlainText("scan_table = [[5], [6]]")

    scan_index = widget._sweep_combo.findData(SWEEP_SCAN_SLOT)
    widget._sweep_combo.setCurrentIndex(scan_index)
    assert widget._program_code.toPlainText() == "scan_table = [[1, 2], [3, 4]]"
    widget._sweep_combo.setCurrentIndex(api_index)
    assert widget._program_code.toPlainText() == "scan_table = [[5], [6]]"


def test_saved_override_round_trips_only_for_same_program(_app):
    from Zou_lab_control.frontend.task_console import _PulseSlotsWidget
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_API_SLOT, SWEEP_SCAN_SLOT

    widget = _PulseSlotsWidget()
    widget.seed_value({
        "program_id": "mot-v1",
        "api": {"a1": 7.5},
        "sweep_kind": SWEEP_API_SLOT,
        "program": "scan_table = [[1], [3]]",
    })
    widget.rebuild(
        api_rows=_api_rows(), scan_rows=_scan_rows(),
        hardware_program="scan_table = [[0, 0]]", program_id="mot-v1")
    value = widget.values_dict()
    assert value["api"]["a1"] == pytest.approx(7.5)
    assert value["sweep_kind"] == SWEEP_API_SLOT
    assert "[3]" in value["program"]

    # A different template with the same opaque handle receives neither saved override.
    widget.seed_value({
        "program_id": "mot-v1",
        "api": {"a1": 99},
        "sweep_kind": SWEEP_API_SLOT,
        "program": "scan_table = [[99]]",
    })
    widget.rebuild(
        api_rows=_api_rows(target="3", current=2.0),
        scan_rows=[("probe_duration", "duration", "3", "ns", "probe duration")],
        hardware_program="scan_table = [[20], [40]]",
        program_id="probe-v2",
    )
    value = widget.values_dict()
    assert value["api"]["a1"] == pytest.approx(2.0)
    assert value["sweep_kind"] == SWEEP_SCAN_SLOT
    assert value["program"] == "scan_table = [[20], [40]]"


def test_blank_hardware_program_uses_kind_aware_semantic_starter(_app):
    widget = _widget(code="")
    code = widget._program_code.toPlainText()
    assert "da_x" in code and "da_y" in code
    assert code.strip()
