from __future__ import annotations

import json

import pytest

from zlc_neutral_atom.installation_config import (
    INSTALLATION_CONFIG_FORMAT,
    InstallationConfigConflict,
    InstallationConfigDocument,
    load_installation_config,
    save_installation_config,
    default_installation_authoring_schema,
)
from zlc_neutral_atom.devices.sequencer.config import RemotePulseInstallationConfig
from zlc_neutral_atom.devices.simulation.config import VirtualInstallationConfig
from zlc_workbench.device_manager.editor_session import (
    DeviceConfigEditorSession,
    form_spec,
)
from zlc_workbench.form_projection import project_authoring_form


def _virtual(seed=7):
    return InstallationConfigDocument.from_parameters("virtual", {"seed": seed})


def _remote(
    host="pulse-host",
    port=18861,
    transport_timeout_seconds=120.0,
):
    return InstallationConfigDocument.from_parameters(
        "remote_pulse",
        {
            "host": host,
            "port": port,
            "transport_timeout_seconds": transport_timeout_seconds,
        },
    )


def test_current_configs_round_trip_through_canonical_json(tmp_path):
    documents = (
        _virtual(seed=19),
        _remote(
            host="pulse-host",
            port=18862,
            transport_timeout_seconds=45.5,
        ),
    )
    for index, document in enumerate(documents):
        path = tmp_path / f"installation-{index}.json"
        digest = save_installation_config(path, document)

        assert path.read_bytes() == document.to_bytes()
        assert digest == document.content_digest
        assert load_installation_config(path) == document

    assert isinstance(documents[0].config, VirtualInstallationConfig)
    assert isinstance(documents[1].config, RemotePulseInstallationConfig)


def test_decoder_is_current_only_and_rejects_legacy_or_ambiguous_json():
    legacy = {
        "camera": {"type": "QCMOSCamera", "params": {}},
        "sequencer": {"type": "RemoteSequencer", "params": {}},
    }
    with pytest.raises(ValueError, match="exactly"):
        InstallationConfigDocument.from_bytes(json.dumps(legacy).encode())

    versioned = {
        "format": INSTALLATION_CONFIG_FORMAT,
        "version": 1,
        "backend": "virtual",
        "parameters": {"seed": 7},
    }
    with pytest.raises(ValueError, match="exactly"):
        InstallationConfigDocument.from_dict(versioned)

    unknown_backend = {
        "format": INSTALLATION_CONFIG_FORMAT,
        "backend": "class_registry",
        "parameters": {},
    }
    with pytest.raises(ValueError, match="unsupported installation backend"):
        InstallationConfigDocument.from_dict(unknown_backend)

    duplicate = (
        '{"format":"zlc_neutral_atom.InstallationConfig",'
        '"backend":"virtual","backend":"remote_pulse",'
        '"parameters":{"seed":7}}'
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        InstallationConfigDocument.from_bytes(duplicate.encode())


def test_field_contracts_reject_non_current_values():
    with pytest.raises(TypeError, match="int"):
        _virtual(seed=True)
    with pytest.raises(ValueError, match="canonical"):
        _remote(host=" pulse-host ")
    with pytest.raises(ValueError, match="maximum"):
        _remote(host="pulse-host", port=65536)
    with pytest.raises(ValueError, match="minimum"):
        _remote(
            host="pulse-host",
            transport_timeout_seconds=0.0,
        )

    extra_parameter = {
        "format": INSTALLATION_CONFIG_FORMAT,
        "backend": "remote_pulse",
        "parameters": {
            "host": "pulse-host",
            "port": 18861,
            "transport_timeout_seconds": 120.0,
            "adapter_class": "module.RemoteSequencer",
        },
    }
    with pytest.raises(ValueError, match="unknown fields"):
        InstallationConfigDocument.from_dict(extra_parameter)


def test_backend_owner_declares_device_manager_form_semantics():
    virtual_schema = default_installation_authoring_schema("virtual")
    assert form_spec("virtual") == project_authoring_form(virtual_schema)
    assert virtual_schema.keys == ("seed",)
    seed = virtual_schema.fields[0]
    assert (
        seed.kind,
        seed.label,
        seed.default,
        seed.required,
        seed.unit,
        seed.minimum,
        seed.maximum,
        seed.allow_blank,
    ) == ("int", "Random seed", 7, False, "", 0, None, True)
    assert seed.description

    remote_schema = default_installation_authoring_schema("remote_pulse")
    assert form_spec("remote_pulse") == project_authoring_form(remote_schema)
    assert remote_schema.keys == (
        "host",
        "port",
        "transport_timeout_seconds",
    )
    host, port, timeout = remote_schema.fields
    assert (host.kind, host.default, host.required) == ("text", "", True)
    assert (port.kind, port.default, port.minimum, port.maximum) == (
        "int",
        18861,
        1,
        65535,
    )
    assert (
        timeout.kind,
        timeout.default,
        timeout.required,
        timeout.unit,
        timeout.maximum,
    ) == ("float", 120.0, True, "s", None)
    assert timeout.minimum == float.fromhex("0x0.0000000000001p-1022")
    assert all(field.description for field in remote_schema.fields)

    current = _remote(
        host="pulse-host",
        port=18862,
        transport_timeout_seconds=45.5,
    )
    assert form_spec(current).default_values() == {
        "host": "pulse-host",
        "port": 18862,
        "transport_timeout_seconds": 45.5,
    }
    editor = DeviceConfigEditorSession(current)
    editor.set_field("port", 18863)
    assert editor.candidate() == _remote(
        host="pulse-host",
        port=18863,
        transport_timeout_seconds=45.5,
    )


def test_expected_digest_is_a_real_compare_and_swap(tmp_path):
    path = tmp_path / "installation.json"
    first = _virtual(seed=1)
    second = _virtual(seed=2)
    third = _virtual(seed=3)

    first_digest = save_installation_config(path, first)
    second_digest = save_installation_config(
        path,
        second,
        expected_digest=first_digest,
    )
    assert second_digest == second.content_digest

    with pytest.raises(InstallationConfigConflict) as caught:
        save_installation_config(
            path,
            third,
            expected_digest=first_digest,
        )
    assert caught.value.expected_digest == first_digest
    assert caught.value.actual_digest == second_digest
    assert load_installation_config(path) == second
    assert not tuple(tmp_path.glob("*.tmp"))


def test_expected_digest_accepts_semantically_identical_reformat(tmp_path):
    path = tmp_path / "installation.json"
    document = _remote(host="pulse-host")
    path.write_text(
        json.dumps(document.to_dict(), indent=2),
        encoding="utf-8",
    )

    digest = document.content_digest
    saved = save_installation_config(
        path,
        document,
        expected_digest=digest,
    )

    assert saved == digest
    assert path.read_bytes() == document.to_bytes()
