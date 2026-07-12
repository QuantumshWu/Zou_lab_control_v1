from __future__ import annotations

import pytest

from Zou_lab_control.neutral_atom import connect
from Zou_lab_control.neutral_atom.device_catalog import InstallationAvailability


def test_session_publishes_catalog_not_raw_devices():
    experiment = connect("virtual")
    try:
        catalog = experiment.device_catalog
        assert not hasattr(experiment, "devices")
        assert catalog.availability is InstallationAvailability.AVAILABLE
        assert catalog.roles() == (
            "camera",
            "laser",
            "monitor_camera",
            "rf",
            "sequencer",
            "trap_array",
        )
        assert not hasattr(catalog, "camera")
        assert not hasattr(catalog.require("camera"), "acquire")
        assert not hasattr(experiment, "camera")
        assert not hasattr(experiment, "sequencer")
    finally:
        experiment.close()


def test_config_swap_reuses_one_runtime_and_publishes_new_generation():
    experiment = connect("virtual")
    try:
        runtime = experiment._require_runtime_services()
        previous_catalog = experiment.device_catalog
        previous_devices = experiment._device_set

        experiment.load_config("virtual")

        current = experiment.device_catalog
        assert experiment._require_runtime_services() is runtime
        assert current.installation_generation > previous_catalog.installation_generation
        assert current.installation_state_revision > previous_catalog.installation_state_revision
        assert current.availability is InstallationAvailability.AVAILABLE
        assert experiment._device_set is not previous_devices
        assert runtime.registry is runtime.fence.devices
    finally:
        experiment.close()


def test_preclose_staging_failure_keeps_complete_old_state(monkeypatch):
    experiment = connect("virtual")
    try:
        runtime = experiment._require_runtime_services()
        previous_catalog = experiment.device_catalog
        previous_devices = experiment._device_set

        def fail_prepare(_devices):
            raise RuntimeError("candidate invalid")

        monkeypatch.setattr(runtime, "prepare_device_set", fail_prepare)
        with pytest.raises(RuntimeError, match="candidate invalid"):
            experiment.load_config("virtual")

        assert experiment.device_catalog is previous_catalog
        assert experiment._device_set is previous_devices
        assert experiment.device_catalog.availability is InstallationAvailability.AVAILABLE
    finally:
        experiment.close()


def test_swapping_is_public_before_first_old_close(monkeypatch):
    experiment = connect("virtual")
    try:
        old_devices = experiment._device_set
        real_close = old_devices.close
        observations = []

        def observe_then_close():
            observations.append(experiment.device_catalog.availability)
            return real_close()

        monkeypatch.setattr(old_devices, "close", observe_then_close)
        experiment.load_config("virtual")
        assert observations == [InstallationAvailability.SWAPPING]
        assert experiment.device_catalog.availability is InstallationAvailability.AVAILABLE
    finally:
        experiment.close()


def test_post_boundary_failure_retains_private_recovery_context(monkeypatch):
    experiment = connect("virtual")
    old_devices = experiment._device_set
    real_close = old_devices.close
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("close failed")
        return real_close()

    monkeypatch.setattr(old_devices, "close", fail_once)
    try:
        with pytest.raises(RuntimeError, match="close failed"):
            experiment.load_config("virtual")

        catalog = experiment.device_catalog
        assert catalog.availability is InstallationAvailability.RECOVERY_REQUIRED
        assert catalog.recovery_status_ref is not None
        assert catalog.roles() == (
            "camera",
            "laser",
            "monitor_camera",
            "rf",
            "sequencer",
            "trap_array",
        )
        with pytest.raises(RuntimeError, match="recovery_required"):
            _ = experiment._device_set
        assert experiment._installation_supervisor._recovery() is not None
        public = experiment._installation_supervisor.catalog_reader.snapshot()
        assert not hasattr(public, "recovery_context")
        assert not hasattr(public, "device_set")
    finally:
        experiment.close()
