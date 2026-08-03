"""Checked-in remote-pulse graph template."""

from zlc_neutral_atom.installation_config import (
    DeviceInstanceConfig,
    InstallationConfigDocument,
)


INSTALLATION_TEMPLATES = {
    "remote_pulse": InstallationConfigDocument(
        (
            DeviceInstanceConfig(
                "sequencer",
                "sequencer",
                "sequencer.remote_pulse",
                {
                    "host": "127.0.0.1",
                    "port": 18861,
                    "transport_timeout_seconds": 120.0,
                },
            ),
        )
    )
}


__all__ = ["INSTALLATION_TEMPLATES"]
