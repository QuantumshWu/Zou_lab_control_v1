"""Composition of qCMOS, Basler MOT camera, and remote pulse hardware."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from zlc_neutral_atom.devices.camera.binding import bind_camera_endpoint
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.camera.contract import CameraAcquisitionMode, CameraAdapter
from zlc_neutral_atom.devices.camera.dcam import DcamCameraAdapter, DcamCameraConfig
from zlc_neutral_atom.devices.camera.endpoint import CameraMonitorEndpoint
from zlc_neutral_atom.devices.camera.monitor import BoundCameraMonitorPort
from zlc_neutral_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
from zlc_neutral_atom.devices.sequencer.installation import (
    bind_remote_sequencer,
    connect_remote_pulse_client,
)
from zlc_neutral_atom.installation import ReadoutApparatusFacts
from zlc_neutral_atom.installation_assets import InstallationAsset, InstallationAssetMap, adapter_kind
from zlc_neutral_atom.installation_plan import InstallationDevicePlan
from zlc_neutral_atom.installation_runtime import _InstallationComposition, _InstallationRuntime, _catalog
from zlc_neutral_atom.runtime.ports import DeviceBroker
from zlc_neutral_atom.runtime.resources import DeviceIdentityEvidenceKind, ResourceArbiter, ResourceKey
from zlc_neutral_atom.runtime.run import RunController
from zlc_pulse import PulseDocument, RemotePulseExecutionClient
from zlc_storage import canonical_digest

from .config import HardwareInstallationConfig
from .qualification import qualify_external_trigger_path


def _asset(role: str, device: object, endpoint: dict[str, object]) -> InstallationAsset:
    identity = canonical_digest(endpoint)
    return InstallationAsset(
        asset_id=f"hardware-{role}",
        role=role,
        resource_key=ResourceKey.parse(f"device/{role}"),
        adapter_kind=adapter_kind(device),
        evidence_kind=DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        expected_identity=f"installation-endpoint:{identity}",
    )


def create_hardware_installation(
    config: HardwareInstallationConfig,
    *,
    required_pulse_document: PulseDocument | None = None,
    device_plan: tuple[InstallationDevicePlan, ...] | None = None,
    remote_client_factory: Callable[..., RemotePulseExecutionClient] | None = None,
    dcam_factory: Callable[[DcamCameraConfig], CameraAdapter] = DcamCameraAdapter,
    pylon_factory: Callable[[PylonCameraConfig], CameraAdapter] = PylonCameraAdapter,
) -> _InstallationComposition:
    """Initialize the complete real graph, including active E0 qualification.

    DeviceManager's explicit Initialize action is the bring-up boundary.  This
    function publishes no runtime unless both physical trigger paths pass E0.
    """

    if not isinstance(config, HardwareInstallationConfig):
        raise TypeError("config must be HardwareInstallationConfig")
    if device_plan is None:
        from .package import INSTALLATION_PACKAGE

        device_plan = INSTALLATION_PACKAGE.device_plan
    connection = connect_remote_pulse_client(
        host=config.pulse_host,
        port=config.pulse_port,
        transport_timeout_seconds=config.pulse_transport_timeout_seconds,
        required_pulse_document=required_pulse_document,
        client_factory=remote_client_factory,
    )
    client = connection.client
    camera: CameraAdapter | None = None
    mot_camera: CameraAdapter | None = None
    broker: DeviceBroker | None = None
    resources: ResourceArbiter | None = None
    try:
        camera = dcam_factory(
            DcamCameraConfig(
                capture_trigger_channels=(config.readout_trigger_lane,),
                exposure_seconds=config.dcam_exposure_seconds,
                readout_speed=config.dcam_readout_speed,
                binning=config.dcam_binning,
                roi_xywh=config.dcam_roi_xywh,
                device_index=config.dcam_device_index,
            )
        )
        if not isinstance(camera, CameraAdapter):
            raise TypeError("dcam_factory returned a non-CameraAdapter")
        mot_camera = pylon_factory(
            PylonCameraConfig(
                serial=config.pylon_serial,
                capture_trigger_channels=(config.mot_trigger_lane,),
                exposure_seconds=config.pylon_exposure_seconds,
                trigger_source=config.pylon_trigger_source,
                roi_xywh=config.pylon_roi_xywh,
                timeout_seconds=config.pylon_timeout_seconds,
            )
        )
        if not isinstance(mot_camera, CameraAdapter):
            raise TypeError("pylon_factory returned a non-CameraAdapter")

        readout_qualification = qualify_external_trigger_path(
            client=client,
            camera=camera,
            trigger_lane=config.readout_trigger_lane,
        )
        mot_qualification = qualify_external_trigger_path(
            client=client,
            camera=mot_camera,
            trigger_lane=config.mot_trigger_lane,
        )
        readout_working_point = camera.capture_working_point()

        devices: dict[str, object] = {
            "sequencer": client,
            "camera": camera,
            "mot_camera": mot_camera,
        }
        assets = InstallationAssetMap(
            (
                _asset(
                    "sequencer",
                    client,
                    {
                        "protocol": "zlc.current-pulse-rpc",
                        "host": config.pulse_host,
                        "port": config.pulse_port,
                    },
                ),
                _asset(
                    "camera",
                    camera,
                    {"sdk": "dcam", "device_index": config.dcam_device_index},
                ),
                _asset(
                    "mot_camera",
                    mot_camera,
                    {"sdk": "pylon", "serial": config.pylon_serial},
                ),
            )
        )
        installation_id = f"installation-{assets.revision[:20]}"
        runtime_instance_id = uuid.uuid4().hex
        broker = DeviceBroker()

        camera_endpoint = CameraMonitorEndpoint(
            camera,
            "camera",
            exact_external_trigger_qualification_digest=readout_qualification,
            acquisition_mode=CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            monitor_acquisition_mode=CameraAcquisitionMode.EXTERNAL_TRIGGERED,
        )
        camera_attestation = bind_camera_endpoint(
            broker,
            assets.require("camera", camera),
            assets.revision,
            camera_endpoint,
        )
        mot_endpoint = CameraMonitorEndpoint(
            mot_camera,
            "mot_camera",
            exact_external_trigger_qualification_digest=mot_qualification,
            acquisition_mode=CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            monitor_acquisition_mode=CameraAcquisitionMode.FREE_RUNNING,
        )
        mot_attestation = bind_camera_endpoint(
            broker,
            assets.require("mot_camera", mot_camera),
            assets.revision,
            mot_endpoint,
        )
        pulse_port = bind_remote_sequencer(
            broker,
            assets.require("sequencer", client),
            assets.revision,
            client,
            endpoint_label=connection.endpoint_label,
            max_blocking_call_seconds=connection.max_blocking_call_seconds,
        )
        catalog = _catalog(
            installation_id,
            runtime_instance_id,
            assets,
            devices,
            device_plan,
        )
        resources = ResourceArbiter()
        runtime = _InstallationRuntime(
            installation_id=installation_id,
            runtime_instance_id=runtime_instance_id,
            catalog=catalog,
            resources=resources,
            broker=broker,
            controller=RunController(resources),
            camera_ports={
                "camera": BoundCapturePort(camera_attestation),
                "mot_camera": BoundCapturePort(mot_attestation),
            },
            camera_monitor_ports={
                "camera": BoundCameraMonitorPort(camera_attestation),
                "mot_camera": BoundCameraMonitorPort(mot_attestation),
            },
            pulse_ports={"sequencer": pulse_port},
            rf_ports={},
            raw_graph=devices,
            close_order=("mot_camera", "camera", "sequencer"),
        )
        apparatus = ReadoutApparatusFacts(
            camera_role="camera",
            sequencer_role="sequencer",
            frame_shape_yx=readout_working_point.frame_shape_yx,
            grid_shape_yx=config.readout_grid_shape_yx,
            site_centers_xy=config.readout_site_centers_xy,
            trigger_channel=config.readout_trigger_lane,
        )
        return _InstallationComposition(
            runtime=runtime,
            readout_apparatus_facts=(apparatus,),
            camera_signal_association_authorities=(
                ("camera", camera_endpoint),
            ),
        )
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        for action in (
            None if broker is None else broker.shutdown,
            None if mot_camera is None else mot_camera.close,
            None if camera is None else camera.close,
            client.close,
        ):
            if action is None:
                continue
            try:
                action()
            except BaseException as error:
                cleanup_errors.append(error)
        if resources is not None:
            try:
                resources.shutdown()
            except BaseException as error:
                cleanup_errors.append(error)
        for error in cleanup_errors:
            primary.add_note(
                "hardware installation startup cleanup also failed: "
                f"{type(error).__name__}: {error}"
            )
        raise


__all__ = ["create_hardware_installation"]
