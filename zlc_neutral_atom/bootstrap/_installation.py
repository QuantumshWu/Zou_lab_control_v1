"""Private typed installation-config dispatch."""

from __future__ import annotations

from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
    RemotePulseInstallationConfig,
    VirtualInstallationConfig,
)
from zlc_neutral_atom.installation_runtime import _InstallationComposition
from zlc_pulse import PulseDocument


def create_installation(
    document: InstallationConfigDocument,
    *,
    required_pulse_document: PulseDocument | None = None,
) -> _InstallationComposition:
    """Compose exactly one of the current typed installation variants."""

    if not isinstance(document, InstallationConfigDocument):
        raise TypeError("document must be InstallationConfigDocument")
    config = document.config
    if isinstance(config, VirtualInstallationConfig):
        if required_pulse_document is not None:
            raise ValueError(
                "required_pulse_document is valid only for remote_pulse"
            )
        from zlc_neutral_atom.devices.simulation.installation import (
            create_virtual_installation,
        )

        return create_virtual_installation(seed=config.seed)
    if isinstance(config, RemotePulseInstallationConfig):
        from zlc_neutral_atom.devices.sequencer.installation import (
            create_remote_pulse_installation,
        )

        return create_remote_pulse_installation(
            host=config.host,
            port=config.port,
            transport_timeout_seconds=config.transport_timeout_seconds,
            required_pulse_document=required_pulse_document,
        )
    raise TypeError("document contains an unknown installation config")


__all__ = ["create_installation"]
