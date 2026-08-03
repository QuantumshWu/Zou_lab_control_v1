"""Checked-in real-apparatus graph template; physical identifiers stay editable."""

from zlc_neutral_atom.installation_config import (
    DeviceInstanceConfig,
    InstallationConfigDocument,
)


INSTALLATION_TEMPLATES = {
    "hardware": InstallationConfigDocument(
        (
            DeviceInstanceConfig(
                "sequencer",
                "sequencer",
                "sequencer.remote_pulse",
                {
                    "host": "127.0.0.1",
                    "port": 18861,
                },
            ),
            DeviceInstanceConfig(
                "camera",
                "camera",
                "camera.dcam",
                {
                    "sequencer_ref": "sequencer",
                    "device_index": 0,
                    "exposure_seconds": 0.02,
                    "readout_speed": 1,
                    "roi_x": None,
                    "roi_y": None,
                    "roi_width": None,
                    "roi_height": None,
                },
            ),
            DeviceInstanceConfig(
                "mot-camera",
                "mot_camera",
                "camera.pylon",
                {
                    "sequencer_ref": "sequencer",
                    "serial": "REQUIRED",
                    "exposure_seconds": 0.005,
                    "trigger_source": "Line1",
                    "roi_x": None,
                    "roi_y": None,
                    "roi_width": None,
                    "roi_height": None,
                },
            ),
        )
    )
}


__all__ = ["INSTALLATION_TEMPLATES"]
