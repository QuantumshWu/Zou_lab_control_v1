"""Compose installed physical facts into Calibration-owned acquisition intent."""

from __future__ import annotations

from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.installation import DeviceRef

from .calibration import GridOrder
from .sitemap import (
    ReadoutGridGeometry,
    SitemapAcquisitionProfile,
)


_DEFAULT_MAXIMUM_SITE_RESIDUAL_PX = 2.0


def build_sitemap_acquisition_profile(
    *,
    grid_shape_yx: tuple[int, int],
    camera_ref: DeviceRef,
    sequencer_ref: DeviceRef,
    camera_port: BoundCapturePort,
    pulse_port: BoundPulsePort,
    trigger_channel: str,
    expected_centers_xy=None,
    maximum_site_residual_px: float | None = None,
) -> SitemapAcquisitionProfile:
    """Bind wiring and a calibration-owned grid to exact live Ports.

    The installation layer supplies only physical wiring.  Grid dimensions and
    optional expected centers are calibration intent; they are validated and
    owned here rather than being smuggled through a camera Device Manager card.
    The composition root supplies already-bound Ports, so this function still
    performs no project-file I/O and cannot silently target another runtime.
    """

    if not isinstance(camera_ref, DeviceRef):
        raise TypeError("camera_ref must be DeviceRef")
    if not isinstance(sequencer_ref, DeviceRef):
        raise TypeError("sequencer_ref must be DeviceRef")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    camera_facts = camera_port.capability.camera_physical_facts
    geometry = ReadoutGridGeometry(
        frame_shape_yx=camera_facts.output_shape_yx,
        spatial_y_axis_id=camera_facts.spatial_y_axis_id,
        spatial_x_axis_id=camera_facts.spatial_x_axis_id,
        coordinate_frame=camera_facts.coordinate_frame,
        grid_shape_yx=grid_shape_yx,
        ordering=GridOrder.ROW_MAJOR,
        expected_centers_xy=expected_centers_xy,
    )
    return SitemapAcquisitionProfile(
        readout_binding=ReadoutBindingKey(camera_ref.instance_id),
        camera_instance_id=camera_ref.instance_id,
        sequencer_instance_id=sequencer_ref.instance_id,
        camera_facts=camera_facts,
        geometry=geometry,
        maximum_site_residual_px=(
            _DEFAULT_MAXIMUM_SITE_RESIDUAL_PX
            if expected_centers_xy is not None and maximum_site_residual_px is None
            else maximum_site_residual_px
        ),
        pulse_target=pulse_port.capability.target,
        trigger_channel=trigger_channel,
    )


__all__ = [
    "build_sitemap_acquisition_profile",
]
