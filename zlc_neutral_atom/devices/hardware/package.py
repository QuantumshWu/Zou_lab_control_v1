"""Real qCMOS + Basler + remote sequencer installation package."""

from __future__ import annotations

from zlc_neutral_atom.installation_package import InstallationPackage
from zlc_neutral_atom.installation_plan import InstallationDevicePlan
from zlc_neutral_atom.installation_runtime import _InstallationComposition
from zlc_pulse import PulseDocument

from .config import (
    HardwareInstallationConfig,
    hardware_authoring_schema,
    hardware_config_from_parameters,
    hardware_config_to_parameters,
)


_DEVICE_PLAN = (
    InstallationDevicePlan(
        "sequencer",
        "sequencer",
        "zlc_pulse.client.RemotePulseExecutionClient",
        "Current remote FPGA pulse execution endpoint",
        ("pulse_host", "pulse_port", "pulse_transport_timeout_seconds"),
    ),
    InstallationDevicePlan(
        "camera",
        "camera",
        "zlc_neutral_atom.devices.camera.dcam.DcamCameraAdapter",
        "Externally triggered qCMOS readout camera",
        (
            "dcam_device_index",
            "dcam_exposure_seconds",
            "dcam_readout_speed",
            "dcam_binning",
            "dcam_roi_x",
            "dcam_roi_y",
            "dcam_roi_width",
            "dcam_roi_height",
            "readout_trigger_lane",
        ),
    ),
    InstallationDevicePlan(
        "mot_camera",
        "camera",
        "zlc_neutral_atom.devices.camera.pylon.PylonCameraAdapter",
        "Basler MOT camera with live and finite triggered acquisition",
        (
            "pylon_serial",
            "pylon_exposure_seconds",
            "pylon_trigger_source",
            "pylon_roi_x",
            "pylon_roi_y",
            "pylon_roi_width",
            "pylon_roi_height",
            "pylon_timeout_seconds",
            "mot_trigger_lane",
        ),
    ),
)


def _compose(
    config: object,
    required_pulse_document: PulseDocument | None,
) -> _InstallationComposition:
    if not isinstance(config, HardwareInstallationConfig):
        raise TypeError("hardware installation package requires HardwareInstallationConfig")
    from .installation import create_hardware_installation

    return create_hardware_installation(
        config,
        required_pulse_document=required_pulse_document,
        device_plan=_DEVICE_PLAN,
    )


INSTALLATION_PACKAGE = InstallationPackage(
    backend="hardware",
    label="Real hardware",
    config_type=HardwareInstallationConfig,
    authoring_schema=hardware_authoring_schema,
    config_from_parameters=hardware_config_from_parameters,
    config_to_parameters=hardware_config_to_parameters,
    device_plan=_DEVICE_PLAN,
    compose=_compose,
    pulse_editor_mode="remote",
)


__all__ = ["INSTALLATION_PACKAGE"]
