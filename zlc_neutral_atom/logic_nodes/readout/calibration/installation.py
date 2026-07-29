"""Compose installed physical facts into Calibration-owned acquisition intent."""

from __future__ import annotations

from pathlib import Path

from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.installation import ReadoutApparatusFacts
from zlc_pulse import bind_pulse_document_target

from .calibration import GridOrder
from .sitemap import (
    ReadoutGridGeometry,
    SitemapAcquisitionProfile,
    load_sitemap_pulse,
)


DEFAULT_MAXIMUM_SITE_RESIDUAL_PX = 2.0


def build_sitemap_acquisition_profile(
    apparatus: ReadoutApparatusFacts,
    *,
    camera_port: BoundCapturePort,
    pulse_port: BoundPulsePort,
    pulses_root: Path,
) -> SitemapAcquisitionProfile:
    """Bind one installed apparatus description to its exact live Ports.

    The installation layer owns only physical wiring and geometry.  This
    Calibration owner selects the project pulse-catalog entry and analysis
    tolerance, while the composition root supplies the already-bound Ports.
    Consequently neither installation dispatch nor a simulated device imports a Logic
    node, and the resulting profile cannot silently target another runtime.
    """

    if not isinstance(apparatus, ReadoutApparatusFacts):
        raise TypeError("apparatus must be ReadoutApparatusFacts")
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
        readout_binding=ReadoutBindingKey(apparatus.camera_role),
        sequencer_role=apparatus.sequencer_role,
        camera_facts=camera_facts,
        geometry=geometry,
        maximum_site_residual_px=DEFAULT_MAXIMUM_SITE_RESIDUAL_PX,
        pulse_document=bind_pulse_document_target(
            load_sitemap_pulse(pulses_root),
            pulse_port.capability.target,
        ),
        trigger_channel=apparatus.trigger_channel,
    )


__all__ = [
    "DEFAULT_MAXIMUM_SITE_RESIDUAL_PX",
    "build_sitemap_acquisition_profile",
]
