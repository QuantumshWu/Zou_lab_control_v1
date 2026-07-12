from __future__ import annotations

import threading

import pytest

from Zou_lab_control.neutral_atom.device_catalog import InstallationAvailability
from Zou_lab_control.neutral_atom.installation import (
    InstallationSupervisor,
    RecoveryStatusRef,
)


class _Devices:
    def __init__(self, **devices):
        self.devices = devices


class _Camera:
    pass


class _Sequencer:
    pass


def _supervisor() -> InstallationSupervisor:
    return InstallationSupervisor(
        _Devices(camera=_Camera(), sequencer=_Sequencer()),
        runtime_authority=object(),
        installation_id="installation-a",
    )


def test_state_transitions_never_publish_partial_bindings():
    supervisor = _supervisor()
    initial = supervisor.snapshot_public()
    assert initial.availability is InstallationAvailability.AVAILABLE
    assert initial.catalog.roles() == ("camera", "sequencer")

    swapping = supervisor._publish_swapping()
    assert swapping.availability is InstallationAvailability.SWAPPING
    assert swapping.catalog.roles() == initial.catalog.roles()
    with pytest.raises(RuntimeError, match="swapping"):
        supervisor._available_device_set()

    recovery = supervisor._publish_recovery_required(
        RecoveryStatusRef("recovery/status/swap-1")
    )
    assert recovery.availability is InstallationAvailability.RECOVERY_REQUIRED
    assert recovery.recovery_status_ref == RecoveryStatusRef(
        "recovery/status/swap-1"
    )

    available = supervisor._publish_available(_Devices(camera=_Camera()))
    assert available.availability is InstallationAvailability.AVAILABLE
    assert available.catalog.roles() == ("camera",)
    assert available.catalog.installation_generation > recovery.catalog.installation_generation
    assert supervisor._available_device_set().devices.keys() == {"camera"}


def test_public_reader_is_gap_free_and_rejects_revision_regression():
    supervisor = _supervisor()
    reader = supervisor.catalog_reader
    initial = reader.snapshot()
    box = {}

    def wait_for_update():
        box["snapshot"] = reader.watch(
            initial.installation_state_revision, timeout=1.0
        )

    thread = threading.Thread(target=wait_for_update)
    thread.start()
    supervisor._publish_swapping()
    thread.join(1.0)

    assert not thread.is_alive()
    assert box["snapshot"].availability is InstallationAvailability.SWAPPING
    latest = reader.snapshot()
    assert reader.watch(
        initial.installation_state_revision, timeout=0.0
    ).installation_state_revision == latest.installation_state_revision
    with pytest.raises(ValueError, match="newer"):
        reader.watch(latest.installation_state_revision + 1, timeout=0.0)
    with pytest.raises(TimeoutError):
        reader.watch(latest.installation_state_revision, timeout=0.0)


def test_public_snapshot_has_no_runtime_or_device_set_capability():
    supervisor = _supervisor()
    public = supervisor.catalog_reader.snapshot()

    assert not hasattr(public, "device_set")
    assert not hasattr(public, "runtime")
    assert not hasattr(public.catalog, "device_set")
    assert not hasattr(public.catalog, "runtime")
    assert set(public.__slots__) == {
        "catalog",
        "availability",
        "recovery_status_ref",
    }
