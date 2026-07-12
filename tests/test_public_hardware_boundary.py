"""Mechanical public-object boundary for installation-owned hardware."""

from __future__ import annotations

import pytest

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom import adapter_sdk, testing


RAW_PUBLIC_NAMES = (
    "BaseDevice",
    "CameraDevice",
    "CommandSequencerBackend",
    "DeviceSet",
    "ManualSequencer",
    "PulseController",
    "QCMOSCamera",
    "QCMOSConfig",
    "RemoteSequencer",
    "SequencerDevice",
    "SequencerService",
    "TrapArrayDevice",
    "VirtualCamera",
    "VirtualSequencer",
    "VirtualTrapArray",
    "bind_pulse",
    "discover_devices",
    "load_devices",
    "register_device_class",
    "run_sequencer_server",
    "serve_runtime_sequencer",
    "triggered_frames",
)


@pytest.mark.parametrize("name", RAW_PUBLIC_NAMES)
def test_ordinary_neutral_atom_umbrella_has_no_raw_hardware(name):
    assert name not in na.__all__
    assert not hasattr(na, name)


def test_adapter_sdk_has_contracts_but_no_concrete_or_registry_escape():
    assert adapter_sdk.CameraDevice is not None
    assert adapter_sdk.CameraBufferOverrun is not None
    assert adapter_sdk.CameraCaptureTerminalRecord is not None
    assert adapter_sdk.CameraFrameRecord is not None
    assert adapter_sdk.SequencerDevice is not None
    for forbidden in (
        "QCMOSCamera",
        "RemoteSequencer",
        "VirtualSequencer",
        "load_devices",
        "register_device_class",
    ):
        assert not hasattr(adapter_sdk, forbidden)


def test_simulation_fakes_require_the_explicit_testing_namespace():
    assert testing.VirtualSequencer is not None
    assert testing.bind_test_pulse is not None
    assert not hasattr(na, "VirtualSequencer")


def test_public_session_exposes_catalog_values_not_raw_devices():
    exp = na.connect("virtual")
    try:
        assert not hasattr(exp, "devices")
        assert exp.device_catalog.roles()
        for forbidden in ("camera", "sequencer", "trap_array", "devices"):
            assert not hasattr(exp.device_catalog, forbidden)
        assert not hasattr(exp, "camera")
        assert not hasattr(exp, "sequencer")
    finally:
        exp.close()
