"""Checked-in virtual apparatus graph template."""

from zlc_neutral_atom.installation_config import (
    DeviceInstanceConfig,
    InstallationConfigDocument,
)


INSTALLATION_TEMPLATES = {
    "virtual": InstallationConfigDocument(
        (
            DeviceInstanceConfig("sequencer", "sequencer", "sequencer.virtual", {}),
            DeviceInstanceConfig(
                "rf",
                "rf",
                "rf.virtual",
                {"sequencer_ref": "sequencer"},
            ),
            DeviceInstanceConfig(
                "camera",
                "camera",
                "camera.virtual_readout",
                {"sequencer_ref": "sequencer", "rf_ref": "rf", "seed": 7},
            ),
            DeviceInstanceConfig(
                "mot-camera",
                "mot_camera",
                "camera.virtual_mot",
                {"sequencer_ref": "sequencer", "seed": 7},
            ),
        )
    )
}


__all__ = ["INSTALLATION_TEMPLATES"]
