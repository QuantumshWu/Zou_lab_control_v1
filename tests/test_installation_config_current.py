from __future__ import annotations

import json

import pytest

from zlc_neutral_atom.installation_config import (
    INSTALLATION_CONFIG_FORMAT,
    InstallationConfigConflict,
    InstallationConfigDocument,
    RemotePulseInstallationConfig,
    VirtualInstallationConfig,
    load_installation_config,
    save_installation_config,
)


def test_current_configs_round_trip_through_canonical_json(tmp_path):
    documents = (
        InstallationConfigDocument.virtual(seed=19),
        InstallationConfigDocument.remote_pulse(
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
    with pytest.raises(TypeError, match="integer"):
        InstallationConfigDocument.virtual(seed=True)
    with pytest.raises(ValueError, match="canonical"):
        InstallationConfigDocument.remote_pulse(host=" pulse-host ")
    with pytest.raises(ValueError, match="at most 65535"):
        InstallationConfigDocument.remote_pulse(host="pulse-host", port=65536)
    with pytest.raises(ValueError, match="positive"):
        InstallationConfigDocument.remote_pulse(
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
    with pytest.raises(ValueError, match="exactly"):
        InstallationConfigDocument.from_dict(extra_parameter)


def test_expected_digest_is_a_real_compare_and_swap(tmp_path):
    path = tmp_path / "installation.json"
    first = InstallationConfigDocument.virtual(seed=1)
    second = InstallationConfigDocument.virtual(seed=2)
    third = InstallationConfigDocument.virtual(seed=3)

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
    document = InstallationConfigDocument.remote_pulse(host="pulse-host")
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
