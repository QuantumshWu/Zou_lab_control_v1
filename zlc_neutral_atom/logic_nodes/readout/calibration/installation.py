"""Compose installed physical facts into Calibration-owned acquisition intent."""

from __future__ import annotations

from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.installation import DeviceRef, ReadoutApparatusFacts

from .calibration import GridOrder
from .sitemap import (
    ReadoutGridGeometry,
    SitemapAcquisitionProfile,
)


_DEFAULT_MAXIMUM_SITE_RESIDUAL_PX = 2.0


def build_sitemap_acquisition_profile(
    apparatus: ReadoutApparatusFacts,
    *,
    camera_ref: DeviceRef,
    sequencer_ref: DeviceRef,
    camera_port: BoundCapturePort,
    pulse_port: BoundPulsePort,
) -> SitemapAcquisitionProfile:
    """Bind one installed apparatus description to its exact live Ports.

    The installation layer owns only physical wiring and geometry.  This
    Calibration owns analysis tolerance and later validates an authored pulse,
    while the composition root supplies the already-bound Ports.  Consequently
    package composition performs no project-file I/O, and a bound operation
    still cannot silently target another runtime.
    """

    if not isinstance(apparatus, ReadoutApparatusFacts):
        raise TypeError("apparatus must be ReadoutApparatusFacts")
    if not isinstance(camera_ref, DeviceRef):
        raise TypeError("camera_ref must be DeviceRef")
    if not isinstance(sequencer_ref, DeviceRef):
        raise TypeError("sequencer_ref must be DeviceRef")
    if camera_ref.instance_id != apparatus.camera_instance_id:
        raise ValueError("camera_ref differs from the installed readout apparatus")
    if sequencer_ref.instance_id != apparatus.sequencer_instance_id:
        raise ValueError("sequencer_ref differs from the installed readout apparatus")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    camera_facts = camera_port.capability.camera_physical_facts
    geometry = ReadoutGridGeometry(
        frame_shape_yx=apparatus.frame_shape_yx,
        spatial_y_axis_id=camera_facts.spatial_y_axis_id,
        spatial_x_axis_id=camera_facts.spatial_x_axis_id,
        coordinate_frame=camera_facts.coordinate_frame,
        grid_shape_yx=apparatus.grid_shape_yx,
        ordering=GridOrder.ROW_MAJOR,
        expected_centers_xy=apparatus.site_centers_xy,
    )
    return SitemapAcquisitionProfile(
        readout_binding=ReadoutBindingKey(camera_ref.instance_id),
        camera_instance_id=camera_ref.instance_id,
        sequencer_instance_id=sequencer_ref.instance_id,
        camera_facts=camera_facts,
        geometry=geometry,
        maximum_site_residual_px=_DEFAULT_MAXIMUM_SITE_RESIDUAL_PX,
        pulse_target=pulse_port.capability.target,
        trigger_channel=apparatus.trigger_channel,
    )


__all__ = [
    "build_sitemap_acquisition_profile",
]
