"""Virtual installation backend package."""

from __future__ import annotations

from zlc_neutral_atom.installation_package import InstallationPackage
from zlc_neutral_atom.installation_plan import InstallationDevicePlan
from zlc_neutral_atom.installation_runtime import _InstallationComposition
from zlc_pulse import PulseDocument

from .config import (
    VirtualInstallationConfig,
    virtual_authoring_schema,
    virtual_config_from_parameters,
    virtual_config_to_parameters,
)


_DEVICE_PLAN = (
    InstallationDevicePlan(
        "sequencer",
        "sequencer",
        "zlc_neutral_atom.devices.simulation.apparatus.VirtualSequencer",
        "In-process pulse target execution",
    ),
    InstallationDevicePlan(
        "rf",
        "rf",
        "zlc_neutral_atom.devices.simulation.apparatus.VirtualRfSource",
        "In-process RF-table source driven by the virtual sequencer",
    ),
    InstallationDevicePlan(
        "camera",
        "camera",
        "zlc_neutral_atom.devices.simulation.apparatus.VirtualCamera",
        "Externally triggered readout camera",
        ("seed",),
    ),
    InstallationDevicePlan(
        "mot_camera",
        "camera",
        "zlc_neutral_atom.devices.simulation.apparatus.VirtualCamera",
        "MOT camera with live and finite triggered acquisition",
        ("seed",),
    ),
)

def _compose(
    config: object,
    required_pulse_document: PulseDocument | None,
) -> _InstallationComposition:
    if not isinstance(config, VirtualInstallationConfig):
        raise TypeError("virtual installation package requires VirtualInstallationConfig")
    if required_pulse_document is not None:
        raise ValueError("required_pulse_document is valid only for remote_pulse")
    from .installation import create_virtual_installation

    return create_virtual_installation(seed=config.seed, device_plan=_DEVICE_PLAN)


INSTALLATION_PACKAGE = InstallationPackage(
    backend="virtual",
    label="Virtual",
    config_type=VirtualInstallationConfig,
    authoring_schema=virtual_authoring_schema,
    config_from_parameters=virtual_config_from_parameters,
    config_to_parameters=virtual_config_to_parameters,
    device_plan=_DEVICE_PLAN,
    compose=_compose,
    default=True,
    pulse_editor_mode="virtual",
)

__all__ = ["INSTALLATION_PACKAGE"]
