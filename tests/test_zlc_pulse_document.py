"""Pulse authoring files use one strict typed current schema."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from conftest import tracked_repo_files

from zlc_pulse import (
    FrozenScanTable,
    OutputDelay,
    PulseDocument,
    load_pulse_document,
    pulse_document_from_tree,
    pulse_document_to_tree,
)
from zlc_pulse.document import PULSE_DOCUMENT_SCHEMA


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("path", tracked_repo_files("pulses/*.json"), ids=lambda path: path.name)
def test_all_shipped_authoring_json_loads_into_current_document(path):
    document = load_pulse_document(path)
    assert isinstance(document, PulseDocument)
    assert pulse_document_from_tree(pulse_document_to_tree(document)) == document
    assert json.loads(path.read_text(encoding="utf-8")) == pulse_document_to_tree(document)
    assert len(document.fingerprint) == 64


def test_save_and_load_use_only_the_typed_current_schema(tmp_path):
    document = load_pulse_document(ROOT / "pulses" / "release_recapture.json")
    destination = document.save(tmp_path / "saved")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == PULSE_DOCUMENT_SCHEMA
    for historical in (
        "version",
        "channels",
        "port_catalog",
        "scan_slots",
        "api_slots",
        "analog_bus_programs",
        "repeat_forever",
        "scan_repeats",
    ):
        assert historical not in payload
    assert load_pulse_document(destination) == document


def test_unknown_payload_schema_is_rejected(tmp_path):
    payload = pulse_document_to_tree(
        load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    )
    payload["schema"] = "unsupported-pulse-document"
    source = tmp_path / "old.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema differs"):
        load_pulse_document(source)


def test_document_values_are_deeply_immutable():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    with pytest.raises(FrozenInstanceError):
        document.name = "changed"
    with pytest.raises(TypeError):
        document.periods[0] = document.periods[0]
    with pytest.raises(FrozenInstanceError):
        document.periods[0].duration = 1


def test_authoring_fields_are_numeric_and_bindings_are_typed():
    duration = load_pulse_document(ROOT / "pulses" / "release_recapture.json")
    assert all(isinstance(period.duration, (int, float)) for period in duration.periods)
    assert duration.scan_parameters[0].field.kind == "duration"
    assert duration.scan_parameters[0].field.period_id == "p3"

    dac = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    assert all(
        isinstance(step.value, int)
        for period in dac.periods
        for step in period.analog_steps
    )
    assert {parameter.field.port for parameter in dac.scan_parameters} == {
        "da_bias_x",
        "da_bias_y",
        "da_bias_z",
    }


def test_equal_numeric_values_have_one_canonical_document_identity():
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    period = document.periods[0]
    integer_period = replace(period, duration=20, unit="ns")
    float_period = replace(period, duration=20.0, unit="ns")
    integer = replace(
        document,
        periods=(integer_period, *document.periods[1:]),
        delays=(OutputDelay("ch11", 0, "ns"),),
    )
    floating = replace(
        document,
        periods=(float_period, *document.periods[1:]),
        delays=(OutputDelay("ch11", -0.0, "ns"),),
    )

    assert integer == floating
    assert integer.fingerprint == floating.fingerprint
    assert FrozenScanTable(("x",), ((0.0,),)).fingerprint == FrozenScanTable(
        ("x",),
        ((-0.0,),),
    ).fingerprint


def test_document_rejects_empty_scan_tables_and_large_half_tick_values():
    with pytest.raises(ValueError, match="at least one row"):
        FrozenScanTable(("x",), ())

    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    period = document.periods[0]
    half_tick_ns = (1_000_000_000_000 + 0.5) * document.time_step_ns
    with pytest.raises(ValueError, match="not frozen"):
        replace(
            document,
            periods=(
                replace(period, duration=half_tick_ns, unit="ns"),
                *document.periods[1:],
            ),
        )
