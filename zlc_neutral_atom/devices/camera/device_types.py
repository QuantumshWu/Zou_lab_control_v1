"""Real qCMOS and Basler camera leaves with active trigger qualification."""

from __future__ import annotations

import math
from typing import Mapping

from zlc_neutral_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_neutral_atom.device_types import (
    CAPABILITY_CAMERA_CAPTURE,
    CAPABILITY_CAMERA_MONITOR,
    CAPABILITY_CAMERA_SIGNAL_ASSOCIATION,
    CAPABILITY_EXTERNAL_TRIGGER_CLIENT,
    CAPABILITY_MOT_FIELD_CAPTURE,
    CAPABILITY_READOUT_BINDING,
    DeviceTypeDescriptor,
)
from zlc_neutral_atom.devices.camera.binding import bind_camera_endpoint
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.camera.contract import CameraAcquisitionMode, CameraAdapter
from zlc_neutral_atom.devices.camera.dcam import DcamCameraAdapter, DcamCameraConfig
from zlc_neutral_atom.devices.camera.endpoint import CameraMonitorEndpoint
from zlc_neutral_atom.devices.camera.monitor import BoundCameraMonitorPort
from zlc_neutral_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
from zlc_neutral_atom.devices.hardware.qualification import qualify_external_trigger_path
from zlc_neutral_atom.installation import ReadoutInstallationBinding
from zlc_neutral_atom.installation_config import DeviceInstanceConfig
from zlc_neutral_atom.runtime.ports import DeviceBroker
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
)
from zlc_pulse import RemotePulseExecutionClient
from zlc_storage import canonical_text


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


def _remote_dependency(dependencies: Mapping[str, object]) -> RemotePulseExecutionClient:
    client = dependencies["sequencer_ref"]
    if not isinstance(client, RemotePulseExecutionClient):
        raise TypeError("real camera requires a remote pulse sequencer")
    return client


def _connect_dcam(
    instance: DeviceInstanceConfig,
    dependencies: Mapping[str, object],
    broker: object,
    _required_pulse_document: object | None,
):
    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    client = _remote_dependency(dependencies)
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
            capture_trigger_channels=(values["trigger_lane"],),
            exposure_seconds=values["exposure_seconds"],
            readout_speed=values["readout_speed"],
            binning=values["binning"],
            roi_xywh=roi,
            device_index=values["device_index"],
        )
    )
    if not isinstance(camera, CameraAdapter):
        raise TypeError("DcamCameraAdapter returned a non-CameraAdapter")
    try:
        qualify_external_trigger_path(
            client=client,
            camera=camera,
            trigger_lane=values["trigger_lane"],
        )
        endpoint = CameraMonitorEndpoint(
            camera,
            instance.instance_id,
            exact_external_trigger_qualified=True,
            acquisition_mode=CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            monitor_acquisition_mode=CameraAcquisitionMode.EXTERNAL_TRIGGERED,
        )
        attestation = bind_camera_endpoint(
            broker,
            instance_id=instance.instance_id,
            identity=_camera_identity(f"dcam-device-index:{values['device_index']}"),
            endpoint=endpoint,
        )
        readout_binding = ReadoutInstallationBinding(
            camera_instance_id=instance.instance_id,
            sequencer_instance_id=instance.parameters["sequencer_ref"],
            trigger_channel=values["trigger_lane"],
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
        CAPABILITY_READOUT_BINDING: readout_binding,
    }, camera.close


def _connect_pylon(
    instance: DeviceInstanceConfig,
    dependencies: Mapping[str, object],
    broker: object,
    _required_pulse_document: object | None,
):
    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    client = _remote_dependency(dependencies)
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
            capture_trigger_channels=(values["trigger_lane"],),
            exposure_seconds=values["exposure_seconds"],
            trigger_source=values["trigger_source"],
            roi_xywh=roi,
        )
    )
    if not isinstance(camera, CameraAdapter):
        raise TypeError("PylonCameraAdapter returned a non-CameraAdapter")
    try:
        qualify_external_trigger_path(
            client=client,
            camera=camera,
            trigger_lane=values["trigger_lane"],
        )
        endpoint = CameraMonitorEndpoint(
            camera,
            instance.instance_id,
            exact_external_trigger_qualified=True,
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
        AuthoringField("sequencer_ref", "text", "Sequencer", "sequencer", True),
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
        AuthoringField(
            "binning",
            "choice",
            "Binning",
            1,
            True,
            choices=tuple(AuthoringChoice(value, str(value)) for value in (1, 2, 4, 8, 16)),
        ),
        *_ROI_FIELDS,
        AuthoringField(
            "trigger_lane",
            "text",
            "Sequencer trigger lane",
            "ch11",
            True,
            description="Installation wiring: FPGA lane connected to the qCMOS external trigger.",
        ),
    )
)
_PYLON_SCHEMA = AuthoringSchema(
    (
        AuthoringField("sequencer_ref", "text", "Sequencer", "sequencer", True),
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
        AuthoringField(
            "trigger_lane",
            "text",
            "Sequencer trigger lane",
            "ch06",
            True,
            description="Installation wiring: FPGA lane connected to the Basler trigger input.",
        ),
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
            CAPABILITY_READOUT_BINDING,
        ),
        (("sequencer_ref", CAPABILITY_EXTERNAL_TRIGGER_CLIENT),),
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
        (("sequencer_ref", CAPABILITY_EXTERNAL_TRIGGER_CLIENT),),
        _connect_pylon,
    ),
)


__all__ = ["DEVICE_TYPES"]
