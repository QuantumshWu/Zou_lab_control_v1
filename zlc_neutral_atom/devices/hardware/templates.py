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
                    "transport_timeout_seconds": 120.0,
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
                    "binning": 1,
                    "roi_x": None,
                    "roi_y": None,
                    "roi_width": None,
                    "roi_height": None,
                    "trigger_lane": "ch11",
                    "grid_rows": 1,
                    "grid_columns": 1,
                    "site_centers_json": "[[0,0]]",
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
                    "timeout_seconds": 2.0,
                    "trigger_lane": "ch06",
                },
            ),
        )
    )
}


__all__ = ["INSTALLATION_TEMPLATES"]
