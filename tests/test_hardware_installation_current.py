"""Current hardware composition contracts.

The old test exercised an obsolete camera-owned E0/trigger-lane binding.  The
current contract keeps pulse endpoint selection in the pulse request and keeps
real camera authoring independent from the sequencer graph.
"""

from __future__ import annotations

from zlc_neutral_atom.device_types import device_type
from zlc_neutral_atom.devices.camera.dcam import DcamCameraConfig
from zlc_neutral_atom.devices.camera.pylon import PylonCameraConfig
from zlc_neutral_atom.devices.hardware.templates import INSTALLATION_TEMPLATES


def test_real_camera_descriptors_have_no_pulse_wiring_fields() -> None:
    for type_id in ("camera.dcam", "camera.pylon"):
        descriptor = device_type(type_id)
        assert "sequencer_ref" not in descriptor.authoring_schema.keys
        assert "trigger_lane" not in descriptor.authoring_schema.keys
        assert "capture_trigger_channels" not in descriptor.authoring_schema.keys
        assert descriptor.requirements == ()


def test_hardware_template_keeps_camera_and_sequencer_as_independent_devices() -> None:
    document = INSTALLATION_TEMPLATES["hardware"]
    camera = next(item for item in document.devices if item.type_id == "camera.dcam")
    mot = next(item for item in document.devices if item.type_id == "camera.pylon")
    assert "sequencer_ref" not in camera.parameters
    assert "sequencer_ref" not in mot.parameters


def test_adapter_configs_contain_camera_local_settings_only() -> None:
    assert not hasattr(DcamCameraConfig, "capture_trigger_channels")
    assert not hasattr(PylonCameraConfig, "capture_trigger_channels")
