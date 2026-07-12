from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from Zou_lab_control.neutral_atom.device_catalog import (
    DeviceCatalogView,
    DeviceHealth,
    DeviceInfo,
    DeviceRef,
    InstallationAvailability,
    unavailable_catalog,
)


def _catalog() -> DeviceCatalogView:
    generation = 3
    return DeviceCatalogView(
        "installation-a",
        generation,
        11,
        7,
        (
            DeviceInfo(
                DeviceRef("installation-a", generation, "sequencer"),
                "sequencer",
                "virtual-sequencer",
                "device/sequencer",
                health=DeviceHealth.HEALTHY,
            ),
            DeviceInfo(
                DeviceRef("installation-a", generation, "camera"),
                "camera",
                "virtual-camera",
                "device/camera",
            ),
        ),
    )


def _walk(value):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)


def test_catalog_is_deterministic_immutable_canonical_data():
    catalog = _catalog()

    assert catalog.roles() == ("camera", "sequencer")
    assert catalog.roles("camera") == ("camera",)
    assert catalog.find("missing") is None
    assert catalog.require("camera").ref.installation_generation == 3
    with pytest.raises(KeyError, match="missing"):
        catalog.require("missing")
    with pytest.raises(TypeError):
        catalog["camera"] = catalog["sequencer"]
    with pytest.raises(FrozenInstanceError):
        catalog["camera"].domain = "other"

    tree = catalog.to_dict()
    assert json.loads(json.dumps(tree, sort_keys=True)) == tree
    assert not any(callable(item) for item in _walk(tree))
    assert list(tree) == [
        "installation_id",
        "installation_generation",
        "installation_state_revision",
        "revision",
        "availability",
        "devices",
        "recovery_status_ref",
    ]


def test_catalog_rejects_cross_generation_and_command_shaped_metadata():
    with pytest.raises(ValueError, match="another catalog generation"):
        DeviceCatalogView(
            "installation-a",
            2,
            1,
            1,
            (
                DeviceInfo(
                    DeviceRef("installation-a", 1, "camera"),
                    "camera",
                    "virtual-camera",
                    "device/camera",
                ),
            ),
        )

    public_names = {
        name
        for owner in (DeviceRef, DeviceInfo, DeviceCatalogView)
        for name in dir(owner)
        if not name.startswith("_")
    }
    assert public_names.isdisjoint(
        {"open", "close", "configure", "arm", "acquire", "prepare", "fire", "abort", "safe"}
    )


def test_unavailable_catalog_preserves_roles_but_mints_new_refs():
    previous = _catalog()
    swapping = unavailable_catalog(
        previous,
        installation_generation=4,
        installation_state_revision=12,
        revision=8,
        availability=InstallationAvailability.SWAPPING,
    )

    assert swapping.roles() == previous.roles()
    assert swapping.availability is InstallationAvailability.SWAPPING
    assert swapping["camera"].ref != previous["camera"].ref
    assert swapping["camera"].health is DeviceHealth.UNAVAILABLE
    assert swapping.recovery_status_ref is None

    recovery = unavailable_catalog(
        swapping,
        installation_generation=5,
        installation_state_revision=13,
        revision=9,
        availability=InstallationAvailability.RECOVERY_REQUIRED,
        recovery_status_ref="recovery/status/abc",
    )
    assert recovery.recovery_status_ref == "recovery/status/abc"
    assert recovery["sequencer"].ref.installation_generation == 5


def test_availability_and_health_cannot_contradict_each_other():
    with pytest.raises(ValueError, match="unavailable health"):
        DeviceInfo(
            DeviceRef("installation-a", 1, "camera"),
            "camera",
            "virtual-camera",
            "device/camera",
            InstallationAvailability.SWAPPING,
            DeviceHealth.UNKNOWN,
        )
    with pytest.raises(ValueError, match="recovery status"):
        DeviceCatalogView(
            "installation-a",
            1,
            1,
            1,
            (),
            availability=InstallationAvailability.RECOVERY_REQUIRED,
        )
