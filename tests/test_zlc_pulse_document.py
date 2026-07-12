"""Current PulseDocument plus the single historical file-load boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from zlc_pulse import (
    PULSE_DOCUMENT_SCHEMA,
    PulseDocument,
    load_pulse_document,
    pulse_document_from_tree,
    pulse_document_to_tree,
)
from zlc_pulse.legacy import (
    pulse_document_from_legacy_sequence,
    pulse_document_from_legacy_table,
)


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "camera_imaging_address_switch.json",
        "imaging_template.json",
        "mot_field_template.json",
        "probe_template.json",
        "pulse_test.json",
        "release_recapture.json",
        "T.json",
    ],
)
def test_all_shipped_authoring_json_loads_into_current_document(name):
    document = load_pulse_document(ROOT / "pulses" / name)
    assert isinstance(document, PulseDocument)
    assert pulse_document_from_tree(pulse_document_to_tree(document)) == document
    assert len(document.fingerprint) == 64


def test_old_v3_files_load_but_save_only_current_schema(tmp_path):
    source = ROOT / "tests" / "fixtures" / "legacy_pulse_v3_release_recapture.json"
    document = load_pulse_document(source)
    assert [slot.name for slot in document.scan_slots] == ["t_off"]

    destination = document.save(tmp_path / "converted")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == PULSE_DOCUMENT_SCHEMA
    assert "version" not in payload
    assert "channels" not in payload
    assert "port_catalog" not in payload
    assert load_pulse_document(destination) == document


def test_synthesized_v1_table_translates_at_load_boundary():
    payload = {
        "schema": "Zou_lab_control.neutral_atom.PulseTableState",
        "version": 1,
        "name": "old pulse",
        "channels": ["trap", "probe", "emCCD"],
        "visible_channels": ["trap", "probe"],
        "x_ns": [0.0, 10.0],
        "periods": [
            {"duration": 20.0, "unit": "us", "states": [1, 0, 0]},
            {"duration": 5.0, "unit": "us", "states": [0, 1, 1]},
        ],
    }
    document = pulse_document_from_legacy_table(payload)
    assert document.name == "old pulse"
    assert document.visible_ports == ("trap", "probe")
    assert len(document.periods) == 2
    assert set(pulse_document_to_tree(document)) == {
        "schema",
        "name",
        "target",
        "time_step_ns",
        "periods",
        "scan_slots",
        "scan_table",
        "scan_code",
        "api_slots",
        "visible_ports",
        "analog_bus_programs",
        "delays",
        "repeat",
        "scan_repeats",
    }


def test_flat_legacy_sequence_without_source_table_is_losslessly_segmented():
    payload = {
        "schema": "Zou_lab_control.neutral_atom.PulseSequence",
        "version": 2,
        "name": "flat",
        "delays": {"camera": 2e-9},
        "repeat_count": 3,
        "repeat_period": 6e-6,
        "repeat_forever": False,
        "pulses": [
            {
                "channel": "camera",
                "start": 1e-6,
                "duration": 2e-6,
                "value": 1,
                "name": "expose",
            }
        ],
        "source_table": None,
    }
    document = pulse_document_from_legacy_sequence(payload)
    assert [period.duration for period in document.periods] == pytest.approx(
        [1e-6, 2e-6, 3e-6]
    )
    assert [period.states for period in document.periods] == [(0,), (1,), (0,)]
    assert document.repeat_start == 0
    assert document.repeat_end == 2
    assert document.repeat_count == 3
    assert dict(document.delays) == {"camera": 2e-9}


def test_current_reader_never_performs_legacy_upgrade():
    old = json.loads(
        (ROOT / "pulses" / "imaging_template.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError):
        pulse_document_from_tree(old)
    old["version"] = 5
    with pytest.raises(ValueError, match="version"):
        pulse_document_from_legacy_table(old)


def test_compiled_runtime_program_is_not_misread_as_an_authoring_document():
    with pytest.raises(ValueError, match="unsupported pulse JSON schema"):
        load_pulse_document(ROOT / "pulses" / "mot_field_template_program.json")


def test_document_values_are_deeply_immutable():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    with pytest.raises(FrozenInstanceError):
        document.name = "changed"
    with pytest.raises(TypeError):
        document.scan_table[0] = ()
