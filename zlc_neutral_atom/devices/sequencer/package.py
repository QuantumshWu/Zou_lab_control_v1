"""Remote pulse installation backend package."""

from __future__ import annotations

from zlc_neutral_atom.installation_package import InstallationPackage
from zlc_neutral_atom.installation_plan import InstallationDevicePlan
from zlc_neutral_atom.installation_runtime import _InstallationComposition
from zlc_pulse import PulseDocument

from .config import (
    RemotePulseInstallationConfig,
    remote_pulse_authoring_schema,
    remote_pulse_config_from_parameters,
    remote_pulse_config_to_parameters,
)


_DEVICE_PLAN = (
    InstallationDevicePlan(
        "sequencer",
        "sequencer",
        "zlc_pulse.client.RemotePulseExecutionClient",
        "Remote pulse execution endpoint",
        ("host", "port", "transport_timeout_seconds"),
    ),
)

def _compose(
    config: object,
    required_pulse_document: PulseDocument | None,
) -> _InstallationComposition:
    if not isinstance(config, RemotePulseInstallationConfig):
        raise TypeError(
            "remote pulse installation package requires "
            "RemotePulseInstallationConfig"
        )
    from .installation import create_remote_pulse_installation

    return create_remote_pulse_installation(
        host=config.host,
        port=config.port,
        transport_timeout_seconds=config.transport_timeout_seconds,
        required_pulse_document=required_pulse_document,
        device_plan=_DEVICE_PLAN,
    )


INSTALLATION_PACKAGE = InstallationPackage(
    backend="remote_pulse",
    label="Remote pulse",
    config_type=RemotePulseInstallationConfig,
    authoring_schema=remote_pulse_authoring_schema,
    config_from_parameters=remote_pulse_config_from_parameters,
    config_to_parameters=remote_pulse_config_to_parameters,
    device_plan=_DEVICE_PLAN,
    compose=_compose,
    pulse_editor_mode="remote",
)

__all__ = ["INSTALLATION_PACKAGE"]
