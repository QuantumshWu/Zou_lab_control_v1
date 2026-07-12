"""Pulse authoring files use one strict typed current schema."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from zlc_pulse import (
    PULSE_DOCUMENT_SCHEMA,
    FrozenScanTable,
    OutputDelay,
    PulseDocument,
    PulseExecutionForm,
    compile_pulse_artifact,
    load_pulse_document,
    pulse_document_from_tree,
    pulse_document_to_tree,
)


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("name", "form", "document_digest", "target_ir_digest", "wire_digest"),
    [
        ("camera_imaging_address_switch.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "94893768bbec6511faa55b52dabec52277e52d3aa6c47c1157ebe56308fce75c", "1824c5f4ea06c6c8a299d8e33da5d60071e2b7998155892aa05e4604808eb5fd", "0ebde69385958428c1ca3a39426f6ea321506573cb61ede1b8827429c4cc472c"),
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE, "3880a5f9925f633f53394c65a230063aa76ca42ce24cb389b577da43dfa3200e", "72c6b9a176b17c8882689e19fea319156c67d3398ccd16f30f9a30ed4a91f073", "d3e386a079a5c9fcb83ed979aa03ec890d8db88477cac22814bca6be066c3017"),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "200b162e877776731230fad1b16f023223ff7bfbed2d0e3d68ff5c22ad83ca4a", "2684cbf2dff424bd24595d6e688d74418ef8c3037aca7be3f5f419864d031c00", "e4a37e156503e67e252587439ea5068f2e92fab9bcf75582897027269b0610f1"),
        ("probe_template.json", PulseExecutionForm.STATIC_ONCE, "a6a283727156ea1ce891c1f0ef1fb9256a6e4490d1e091a39269058e42eddc4d", "d60570632f4c840e49d0e9270a10ee7fb073667c6519dc9aef1d8dfdf37b7e80", "b80072f49f39df2e5828572ddc49d356435dd5152a84cd662b83ed53d8144cab"),
        ("release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "e1d02cc83c7e668ac38578e6ef36b23853f28e74f63bcc503a64e359fb63ac19", "ec0b8ae0ad75e12066e1e4c519140d370e5091fbdb591bce53ea5415c7ab036d", "1e58a2f60b09836e581b9c2c11e8c90f81ea490bdcb98c92875a215df5f17b4b"),
    ],
)
def test_converted_assets_keep_the_frozen_physical_program(
    name, form, document_digest, target_ir_digest, wire_digest
):
    document = load_pulse_document(ROOT / "pulses" / name)
    artifact = compile_pulse_artifact(document, clock_hz=50e6, execution_form=form)
    assert document.fingerprint == document_digest
    assert artifact.target_ir.fingerprint == target_ir_digest
    assert artifact.wire_image.digest == wire_digest


@pytest.mark.parametrize(
    "name",
    [
        "camera_imaging_address_switch.json",
        "imaging_template.json",
        "mot_field_template.json",
        "probe_template.json",
        "release_recapture.json",
    ],
)
def test_all_shipped_authoring_json_loads_into_current_document(name):
    path = ROOT / "pulses" / name
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


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": "Zou_lab_control.neutral_atom.PulseTableState", "version": 4},
        {"schema": "zlc_pulse.PulseDocument/v1", "name": "superseded-current-model"},
        {"schema": "Zou_lab_control.neutral_atom.RuntimeSequenceProgram", "version": 4},
    ],
)
def test_historical_or_superseded_payloads_are_rejected(payload, tmp_path):
    source = tmp_path / "old.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field set"):
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
