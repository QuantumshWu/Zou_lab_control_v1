"""Narrow application-host adapter for Camera Measurement commands.

The generic Workbench owns live-slot construction, attachment, notification,
failure cleanup, and final publication.  Camera Measurement owns only the two
facts that a generic host cannot derive: which prepared command branch it was
given and which one-shot method accepts the host's live-output factory.

This optional leaf is imported explicitly by the composition root.  It is not
re-exported by the headless capability package.
"""

from __future__ import annotations

from .finite import PreparedFiniteCameraMeasurement
from .monitor import PreparedLiveCameraMeasurement


def start_camera_measurement_command(
    command: PreparedLiveCameraMeasurement | PreparedFiniteCameraMeasurement,
    live_output_host,
):
    """Start Camera's live/finite command using one generic live-output host.

    The Workbench owns slot construction and cleanup.  Camera owns whether the
    prepared command needs a monitor slot, a finite preview slot, or no slot.
    """

    factory = getattr(live_output_host, "factory", None)
    if not callable(factory):
        raise TypeError("Camera start requires a live-output host")
    live_factory = factory(output_owner=command)
    if isinstance(command, PreparedLiveCameraMeasurement):
        return command.start_with_view(factory=live_factory)
    if not isinstance(command, PreparedFiniteCameraMeasurement):
        raise TypeError("Camera command has another type")
    if command.live_preview_output_name is None:
        return command.start()
    return command.start_with_preview(factory=live_factory)


__all__ = ["start_camera_measurement_command"]
