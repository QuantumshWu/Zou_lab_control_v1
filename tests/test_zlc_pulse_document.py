"""Pulse authoring files use one strict current schema."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from zlc_pulse import (
    PULSE_DOCUMENT_SCHEMA,
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
        ("camera_imaging_address_switch.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "1f5fe38b2fa3d094c7aa6f2252e114d6af0882348cadcc0b5cd55d84f125177c", "1c8f05d640be640f22a72a516ddb2d6c1d08274cdf9c640b160328e0fd042d52", "9d71e0ac3e088fe636e86a9ff4ddee637423683c7d940c3b4330a3c4c1738306"),
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE, "954d443d926a8ff9659c4a76a07a9e42190572faa656cc106f0d79cacdd1b3c0", "3b49211b15c50d0cbdb9f61e35dc8594070c1d6fe8daca058fb18e380a47e67d", "3f3100880aee3be52141ffc1cdda477db38644835ecd6deb361c5e9fcde7c3cb"),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "67ba10682b8c934a52f4ee269639b33934c074fd865723df11d3b30285743bc5", "36934f9e37e3ed785d4144e37b520e2d52e36229aa604229d3eeebc3589dad26", "7e32834207c74fd646e4f9012ae5fc0aceec12dec9c5a5546882a43784b1420a"),
        ("probe_template.json", PulseExecutionForm.STATIC_ONCE, "13030788ec61c8d62eeff181c9beddf892fd2eb25cfe8218bd0cc8c53a662238", "376609a880f39cc65f9b928e38fdaeaa3ee0355d376330f83ad941b369575564", "cc3ff1fc0bb62d7bb83aca831b75fd25cd2d4d7f7e7c0a9a5886d0cd7516375d"),
        ("release_recapture.json", PulseExecutionForm.STATIC_REFERENCE_POINT, "b662555d746d2ca5d72069188d9bd0d0f13805c689c659b1b18a177331bd43c0", "0bacd7db575bb5e830ccb12872c13fb3231bb58be8f2c93e389a58abeee054a7", "64c5c5c1172b6c1dfdc9f2f40bcec1ef2b24719c877027e97881dad1cc11a537"),
    ],
)
def test_converted_tracked_assets_keep_the_frozen_current_compilation(
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


def test_save_and_load_use_only_the_current_schema(tmp_path):
    document = load_pulse_document(ROOT / "pulses" / "release_recapture.json")
    destination = document.save(tmp_path / "saved")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == PULSE_DOCUMENT_SCHEMA
    assert "version" not in payload
    assert "channels" not in payload
    assert "port_catalog" not in payload
    assert load_pulse_document(destination) == document


@pytest.mark.parametrize(
    "payload",
    [
        {
        "schema": "Zou_lab_control.neutral_atom.PulseTableState",
        "version": 4,
        },
        {
            "schema": "Zou_lab_control.neutral_atom.RuntimeSequenceProgram",
            "version": 4,
        },
    ],
)
def test_historical_or_compiled_payloads_are_rejected(payload, tmp_path):
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
