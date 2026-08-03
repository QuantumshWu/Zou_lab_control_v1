from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from zlc_data import AxisId, CoordinateFrameId
from zlc_neutral_atom.devices.camera.contract import (
    CameraPhysicalFacts,
    ReadoutBindingKey,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import GridOrder
from zlc_neutral_atom.logic_nodes.readout.calibration.sitemap import (
    ReadoutGridGeometry,
    SitemapAcquisitionProfile,
    load_sitemap_pulse,
)
from zlc_pulse import PulseTarget


def test_custom_sitemap_pulse_is_rebound_to_the_live_profile_target() -> None:
    base = load_sitemap_pulse(Path(__file__).resolve().parents[1] / "pulses")
    y_axis = AxisId("camera-y")
    x_axis = AxisId("camera-x")
    frame = CoordinateFrameId("camera-output")
    camera_facts = CameraPhysicalFacts(
        camera_identity="camera-id",
        sensor_identity="sensor-id",
        optical_path="imaging",
        capture_trigger_channels=("ch11",),
        sensor_shape_yx=(8, 8),
        roi_origin_yx=(0, 0),
        roi_shape_yx=(8, 8),
        binning_yx=(1, 1),
        spatial_y_axis_id=y_axis,
        spatial_x_axis_id=x_axis,
        coordinate_frame=frame,
        dtype=np.dtype("<u2"),
        count_unit="count",
        exposure_seconds=0.005,
        required_external_trigger_interval_seconds=0.0,
        external_trigger_integration_start_offset_seconds=0.0,
        gain=1.0,
        readout_mode="virtual",
    )
    geometry = ReadoutGridGeometry(
        frame_shape_yx=(8, 8),
        spatial_y_axis_id=y_axis,
        spatial_x_axis_id=x_axis,
        coordinate_frame=frame,
        grid_shape_yx=(1, 1),
        ordering=GridOrder.ROW_MAJOR,
        expected_centers_xy=np.asarray([[4.0, 4.0]]),
    )
    profile = SitemapAcquisitionProfile(
        readout_binding=ReadoutBindingKey("camera"),
        camera_instance_id="camera-instance",
        sequencer_instance_id="sequencer-instance",
        camera_facts=camera_facts,
        geometry=geometry,
        maximum_site_residual_px=1.0,
        pulse_target=base.target,
        trigger_channel="ch11",
    )
    authored_target = PulseTarget(
        base.target.raw_lanes,
        tuple(
            replace(port, key="camera_trigger") if port.key == "ch11" else port
            for port in base.target.ports
        ),
    )
    authored = replace(
        base,
        target=authored_target,
        visible_ports=tuple(
            "camera_trigger" if key == "ch11" else key
            for key in base.visible_ports
        ),
    )

    configured = profile.configured_document_for_repeats(
        2,
        reference_exposure_s=0.020,
        readout_exposure_s=0.005,
        pulse_document=authored,
    )

    assert configured.target is base.target
    assert "ch11" in configured.target.by_key
    assert configured.visible_ports == base.visible_ports


def test_calibration_profile_can_discover_centers_without_installation_prior() -> None:
    base = load_sitemap_pulse(Path(__file__).resolve().parents[1] / "pulses")
    y_axis = AxisId("camera-y")
    x_axis = AxisId("camera-x")
    frame = CoordinateFrameId("camera-output")
    camera_facts = CameraPhysicalFacts(
        camera_identity="camera-id",
        sensor_identity="sensor-id",
        optical_path="imaging",
        capture_trigger_channels=("ch11",),
        sensor_shape_yx=(8, 8),
        roi_origin_yx=(0, 0),
        roi_shape_yx=(8, 8),
        binning_yx=(1, 1),
        spatial_y_axis_id=y_axis,
        spatial_x_axis_id=x_axis,
        coordinate_frame=frame,
        dtype=np.dtype("<u2"),
        count_unit="count",
        exposure_seconds=0.005,
        required_external_trigger_interval_seconds=0.0,
        external_trigger_integration_start_offset_seconds=0.0,
        gain=1.0,
        readout_mode="virtual",
    )
    profile = SitemapAcquisitionProfile(
        readout_binding=ReadoutBindingKey("camera"),
        camera_instance_id="camera-instance",
        sequencer_instance_id="sequencer-instance",
        camera_facts=camera_facts,
        geometry=ReadoutGridGeometry(
            frame_shape_yx=(8, 8),
            spatial_y_axis_id=y_axis,
            spatial_x_axis_id=x_axis,
            coordinate_frame=frame,
            grid_shape_yx=(1, 1),
            ordering=GridOrder.ROW_MAJOR,
        ),
        maximum_site_residual_px=None,
        pulse_target=base.target,
        trigger_channel="ch11",
    )

    request = profile.analysis_request(AxisId("readout-event"))

    assert request.expected_centers_xy is None
    assert request.maximum_site_residual_px is None
