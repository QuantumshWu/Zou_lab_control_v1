"""Real qCMOS and Basler camera leaves.

Camera leaves configure only camera-local settings.  Pulse endpoint selection
belongs to the pulse/measurement request; no real camera connection resolves or
stores an FPGA trigger lane.
"""

from __future__ import annotations

import math
from typing import Mapping

from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_neutral_atom.device_types import (
    CAPABILITY_CAMERA_CAPTURE,
    CAPABILITY_CAMERA_MONITOR,
    CAPABILITY_CAMERA_SIGNAL_ASSOCIATION,
    CAPABILITY_MOT_FIELD_CAPTURE,
    DeviceTypeDescriptor,
)
from zlc_neutral_atom.devices.camera.binding import bind_camera_endpoint
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.camera.contract import CameraAcquisitionMode, CameraAdapter
from zlc_neutral_atom.devices.camera.dcam import DcamCameraAdapter, DcamCameraConfig
from zlc_neutral_atom.devices.camera.endpoint import CameraMonitorEndpoint
from zlc_neutral_atom.devices.camera.monitor import BoundCameraMonitorPort
from zlc_neutral_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
from zlc_neutral_atom.installation_config import DeviceInstanceConfig
from zlc_neutral_atom.runtime.ports import DeviceBroker
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
)


def _optional_roi(values: tuple[object, ...], field: str):
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{field} requires all four ROI values or none")
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field}[{index}] must be int")
        minimum = 0 if index < 2 else 1
        if value < minimum:
            raise ValueError(f"{field}[{index}] must be at least {minimum}")
        result.append(value)
    return tuple(result)


def _camera_identity(value: str) -> PhysicalDeviceIdentity:
    return PhysicalDeviceIdentity(
        value,
        DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
    )


def _connect_dcam(
    instance: DeviceInstanceConfig,
    dependencies: Mapping[str, object],
    broker: object,
    _required_pulse_document: object | None,
):
    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    values = _DCAM_SCHEMA.freeze(instance.parameters)
    roi = _optional_roi(
        (
            values["roi_x"],
            values["roi_y"],
            values["roi_width"],
            values["roi_height"],
        ),
        "dcam_roi",
    )
    camera = DcamCameraAdapter(
        DcamCameraConfig(
            exposure_seconds=values["exposure_seconds"],
            readout_speed=values["readout_speed"],
            binning=1,
            roi_xywh=roi,
            device_index=values["device_index"],
        )
    )
    if not isinstance(camera, CameraAdapter):
        raise TypeError("DcamCameraAdapter returned a non-CameraAdapter")
    try:
        endpoint = CameraMonitorEndpoint(
            camera,
            instance.instance_id,
            acquisition_mode=CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            monitor_acquisition_mode=CameraAcquisitionMode.EXTERNAL_TRIGGERED,
        )
        attestation = bind_camera_endpoint(
            broker,
            instance_id=instance.instance_id,
            identity=_camera_identity(f"dcam-device-index:{values['device_index']}"),
            endpoint=endpoint,
        )
    except BaseException as primary:
        try:
            camera.close()
        except BaseException as error:
            primary.add_note(f"qCMOS close also failed: {error}")
        raise
    return {
        CAPABILITY_CAMERA_CAPTURE: BoundCapturePort(attestation),
        CAPABILITY_CAMERA_MONITOR: BoundCameraMonitorPort(attestation),
        CAPABILITY_CAMERA_SIGNAL_ASSOCIATION: endpoint,
    }, camera.close


def _connect_pylon(
    instance: DeviceInstanceConfig,
    dependencies: Mapping[str, object],
    broker: object,
    _required_pulse_document: object | None,
):
    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    values = _PYLON_SCHEMA.freeze(instance.parameters)
    roi = _optional_roi(
        (
            values["roi_x"],
            values["roi_y"],
            values["roi_width"],
            values["roi_height"],
        ),
        "pylon_roi",
    )
    camera = PylonCameraAdapter(
        PylonCameraConfig(
            serial=values["serial"],
            exposure_seconds=values["exposure_seconds"],
            trigger_source=values["trigger_source"],
            roi_xywh=roi,
        )
    )
    if not isinstance(camera, CameraAdapter):
        raise TypeError("PylonCameraAdapter returned a non-CameraAdapter")
    try:
        endpoint = CameraMonitorEndpoint(
            camera,
            instance.instance_id,
            acquisition_mode=CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            monitor_acquisition_mode=CameraAcquisitionMode.FREE_RUNNING,
        )
        attestation = bind_camera_endpoint(
            broker,
            instance_id=instance.instance_id,
            identity=_camera_identity(f"pylon-serial:{values['serial']}"),
            endpoint=endpoint,
        )
    except BaseException as primary:
        try:
            camera.close()
        except BaseException as error:
            primary.add_note(f"Basler close also failed: {error}")
        raise
    capture = BoundCapturePort(attestation)
    return {
        CAPABILITY_CAMERA_CAPTURE: capture,
        CAPABILITY_CAMERA_MONITOR: BoundCameraMonitorPort(attestation),
        CAPABILITY_MOT_FIELD_CAPTURE: capture,
    }, camera.close


_POSITIVE = math.nextafter(0.0, math.inf)
_ROI_FIELDS = (
    AuthoringField("roi_x", "int", "ROI x", None, False, minimum=0, allow_blank=True),
    AuthoringField("roi_y", "int", "ROI y", None, False, minimum=0, allow_blank=True),
    AuthoringField(
        "roi_width", "int", "ROI width", None, False, minimum=1, allow_blank=True
    ),
    AuthoringField(
        "roi_height", "int", "ROI height", None, False, minimum=1, allow_blank=True
    ),
)
_DCAM_SCHEMA = AuthoringSchema(
    (
        AuthoringField("device_index", "int", "qCMOS device index", 0, True, minimum=0),
        AuthoringField(
            "exposure_seconds",
            "float",
            "qCMOS exposure",
            0.02,
            True,
            unit="s",
            minimum=_POSITIVE,
        ),
        AuthoringField("readout_speed", "int", "Readout speed", 1, True, minimum=1),
        *_ROI_FIELDS,
    )
)
_PYLON_SCHEMA = AuthoringSchema(
    (
        AuthoringField("serial", "text", "Basler serial", "REQUIRED", True),
        AuthoringField(
            "exposure_seconds",
            "float",
            "Basler exposure",
            0.005,
            True,
            unit="s",
            minimum=_POSITIVE,
        ),
        AuthoringField("trigger_source", "text", "Trigger source", "Line1", True),
        *_ROI_FIELDS,
    )
)


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "camera.dcam",
        "camera",
        "Hamamatsu qCMOS",
        _DCAM_SCHEMA,
        (
            CAPABILITY_CAMERA_CAPTURE,
            CAPABILITY_CAMERA_MONITOR,
            CAPABILITY_CAMERA_SIGNAL_ASSOCIATION,
        ),
        (),
        _connect_dcam,
    ),
    DeviceTypeDescriptor(
        "camera.pylon",
        "camera",
        "Basler camera",
        _PYLON_SCHEMA,
        (
            CAPABILITY_CAMERA_CAPTURE,
            CAPABILITY_CAMERA_MONITOR,
            CAPABILITY_MOT_FIELD_CAPTURE,
        ),
        (),
        _connect_pylon,
    ),
)


__all__ = ["DEVICE_TYPES"]
