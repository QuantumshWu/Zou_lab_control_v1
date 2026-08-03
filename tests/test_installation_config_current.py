from __future__ import annotations

import json

import pytest

import zlc_neutral_atom.device_types as device_types
from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_neutral_atom.device_types import (
    DeviceTypeDescriptor,
    validate_installation_graph,
)
from zlc_neutral_atom.installation_config import (
    INSTALLATION_CONFIG_FORMAT,
    DeviceInstanceConfig,
    InstallationConfigDocument,
    discover_installation_templates,
    installation_template,
    load_installation_config,
    save_installation_config,
)


def test_ordered_graph_round_trips_as_ordinary_human_readable_json(tmp_path):
    documents = (
        installation_template("virtual", seed=19),
        installation_template(
            "remote_pulse",
            host="pulse-host",
            port=18862,
        ),
        installation_template("hardware"),
    )

    assert discover_installation_templates() == (
        "hardware",
        "remote_pulse",
        "virtual",
    )
    for index, document in enumerate(documents):
        path = tmp_path / f"installation-{index}.json"
        assert save_installation_config(path, document) is None
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == document.to_dict()
        assert tuple(row["instance_id"] for row in payload["devices"]) == tuple(
            item.instance_id for item in document.devices
        )
        assert load_installation_config(path) == document
        assert not tuple(tmp_path.glob("*.tmp"))


def test_template_override_is_schema_driven_without_backend_dispatch():
    document = installation_template("virtual", seed=23)
    seeded = tuple(
        item.parameters["seed"]
        for item in document.devices
        if "seed" in item.parameters
    )
    assert seeded == (23, 23)


def test_real_hardware_template_keeps_calibration_geometry_out_of_devices():
    document = installation_template("hardware")
    by_id = {item.instance_id: item for item in document.devices}
    camera = by_id["camera"].parameters
    mot_camera = by_id["mot-camera"].parameters
    sequencer = by_id["sequencer"].parameters

    assert {
        "grid_rows",
        "grid_columns",
        "site_centers_json",
        "binning",
        "trigger_lane",
    }.isdisjoint(camera)
    assert "trigger_lane" not in mot_camera
    assert "timeout_seconds" not in mot_camera
    assert "transport_timeout_seconds" not in sequencer
    with pytest.raises(ValueError, match="has no fields"):
        installation_template("virtual", backend="remote")


def test_decoder_is_current_only_and_rejects_legacy_or_ambiguous_json(tmp_path):
    legacy = {
        "format": INSTALLATION_CONFIG_FORMAT,
        "backend": "virtual",
        "parameters": {"seed": 7},
    }
    with pytest.raises(ValueError, match="exactly"):
        InstallationConfigDocument.from_dict(legacy)

    unknown_format = {
        "format": "legacy.installation",
        "devices": [],
    }
    with pytest.raises(ValueError, match="unsupported"):
        InstallationConfigDocument.from_dict(unknown_format)

    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"format":"zlc.installation","devices":[],"devices":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_installation_config(path)


def test_graph_rejects_duplicate_unknown_missing_and_wrong_capability():
    with pytest.raises(ValueError, match="instance_id"):
        InstallationConfigDocument(
            (
                DeviceInstanceConfig("same", "a", "sequencer.virtual", {}),
                DeviceInstanceConfig("same", "b", "sequencer.virtual", {}),
            )
        )
    with pytest.raises(ValueError, match="roles"):
        InstallationConfigDocument(
            (
                DeviceInstanceConfig("a", "same", "sequencer.virtual", {}),
                DeviceInstanceConfig("b", "same", "sequencer.virtual", {}),
            )
        )

    unknown = InstallationConfigDocument(
        (DeviceInstanceConfig("mystery", "mystery", "device.unknown", {}),)
    )
    with pytest.raises(ValueError, match="unknown device type"):
        validate_installation_graph(unknown)

    missing = InstallationConfigDocument(
        (
            DeviceInstanceConfig(
                "rf",
                "rf",
                "rf.virtual",
                {"sequencer_ref": "missing"},
            ),
        )
    )
    with pytest.raises(ValueError, match="references missing"):
        validate_installation_graph(missing)

    wrong = InstallationConfigDocument(
        (
            DeviceInstanceConfig("mot", "mot", "camera.virtual_mot", {
                "sequencer_ref": "mot",
                "seed": 7,
            }),
        )
    )
    with pytest.raises(ValueError, match="does not provide"):
        validate_installation_graph(wrong)


def test_preflight_detects_cycles_without_calling_a_factory(monkeypatch):
    calls = []

    def factory(*_args):
        calls.append("factory")
        return {"test.link": object()}, lambda: None

    descriptor = DeviceTypeDescriptor(
        type_id="test.link",
        domain="test",
        label="Test link",
        authoring_schema=AuthoringSchema(
            (AuthoringField("peer", "text", "Peer", "", True),)
        ),
        capabilities=("test.link",),
        requirements=(("peer", "test.link"),),
        factory=factory,
    )
    monkeypatch.setattr(
        device_types,
        "discover_device_types",
        lambda: (descriptor,),
    )
    document = InstallationConfigDocument(
        (
            DeviceInstanceConfig("a", "a", "test.link", {"peer": "b"}),
            DeviceInstanceConfig("b", "b", "test.link", {"peer": "a"}),
        )
    )

    with pytest.raises(ValueError, match="cycle"):
        validate_installation_graph(document)
    assert calls == []


def test_leaf_schema_rejects_unknown_or_invalid_parameters():
    invalid = InstallationConfigDocument(
        (
            DeviceInstanceConfig(
                "sequencer",
                "sequencer",
                "sequencer.remote_pulse",
                {
                    "host": "pulse-host",
                    "port": 0,
                },
            ),
        )
    )
    with pytest.raises(ValueError, match="minimum"):
        validate_installation_graph(invalid)

    extra = InstallationConfigDocument(
        (
            DeviceInstanceConfig(
                "sequencer",
                "sequencer",
                "sequencer.virtual",
                {"backend": "legacy"},
            ),
        )
    )
    with pytest.raises(ValueError, match="unknown fields"):
        validate_installation_graph(extra)
