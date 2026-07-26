"""Composition of the installation-owned virtual apparatus."""

from __future__ import annotations

import uuid

from fpga.pulse_streamer.host.image import DEFAULT_CONFIG_PATH, default_clock_hz
from zlc_neutral_atom.devices.camera.contract import CameraAcquisitionMode
from zlc_neutral_atom.installation_assets import (
    InstallationAsset,
    InstallationAssetMap,
)
from zlc_neutral_atom.installation_runtime import (
    _InstallationComposition,
    _InstallationRuntime,
    _catalog,
    _identity_for,
)
from zlc_neutral_atom.devices.camera.endpoint import CameraCaptureEndpoint, CameraMonitorEndpoint
from zlc_neutral_atom.devices.camera.binding import bind_camera_endpoint
from zlc_neutral_atom.devices.simulation.apparatus import VirtualAtomArray, VirtualCamera, VirtualMotFrameSource, VirtualRfSource, VirtualSequencer
from zlc_neutral_atom.devices.simulation.rf_endpoint import VirtualRfTableEndpoint
from zlc_neutral_atom.devices.simulation.sequencer_endpoint import VirtualSequencerExecutionEndpoint
from zlc_neutral_atom.installation_plan import InstallationDevicePlan
from zlc_neutral_atom.installation import ReadoutApparatusFacts
from zlc_neutral_atom.devices.rf import BoundRfTablePort
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.camera.monitor import BoundCameraMonitorPort
from zlc_neutral_atom.runtime.ports import BoundDevice, DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import ResourceArbiter
from zlc_neutral_atom.runtime.run import RunController
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_pulse import PORT_DAC, PORT_DIGITAL, PulseTarget, PulseTargetManifest, load_deployed_pulse_target, pulse_target_manifest
from zlc_storage import canonical_digest

# Installation wiring, not a friendly simulator alias.  These are the physical
# lanes used by the checked-in deployed PulseTarget and by the real apparatus.
_COOLING_CHANNELS = ("ch00", "ch01")
_PROBE_CHANNELS = ("ch03",)
_TRAP_CHANNELS = ("ch09",)
_READOUT_CAMERA_TRIGGER_CHANNELS = ("ch11",)
_MOT_CAMERA_TRIGGER_CHANNELS = ("ch06",)
_VIRTUAL_MOT_COIL_PORTS = {
    "da_x": "da_bias_x",
    "da_y": "da_bias_y",
    "da_z": "da_bias_z",
}


def _virtual_imaging_geometry() -> tuple[
    tuple[int, int],
    tuple[int, int],
    tuple[tuple[float, float], ...],
]:
    """Return the one low-level frame, grid, and site-center geometry."""

    grid_shape_yx = (5, 7)
    frame_shape_yx = (96, 128)
    spacing_px = 9.0
    rows, columns = grid_shape_yx
    height, width = frame_shape_yx
    origin_x = (width - (columns - 1) * spacing_px) / 2.0
    origin_y = (height - (rows - 1) * spacing_px) / 2.0
    return (
        frame_shape_yx,
        grid_shape_yx,
        tuple(
            (origin_x + column * spacing_px, origin_y + row * spacing_px)
            for row in range(rows)
            for column in range(columns)
        ),
    )


def _deployed_target() -> PulseTarget:
    """Validate and retain the pulse owner's deployed topology without projection."""

    target = load_deployed_pulse_target()
    required = {
        *_COOLING_CHANNELS,
        *_PROBE_CHANNELS,
        *_TRAP_CHANNELS,
        *_READOUT_CAMERA_TRIGGER_CHANNELS,
        *_MOT_CAMERA_TRIGGER_CHANNELS,
        *_VIRTUAL_MOT_COIL_PORTS.values(),
    }
    missing = tuple(sorted(required.difference(target.by_key)))
    if missing:
        raise RuntimeError(
            f"deployed PulseTarget is missing virtual installation ports {missing}"
        )
    for key in sorted(required.difference(_VIRTUAL_MOT_COIL_PORTS.values())):
        port = target.by_key[key]
        if port.kind != PORT_DIGITAL or port.lanes != (key,):
            raise RuntimeError(
                f"virtual installation wiring {key!r} is not a one-lane digital port"
            )
    for key in _VIRTUAL_MOT_COIL_PORTS.values():
        port = target.by_key[key]
        if port.kind != PORT_DAC or port.latch_clock is None:
            raise RuntimeError(
                f"virtual MOT wiring {key!r} is not a latched DAC port"
            )
    return target


def _virtual_target_manifest(target: PulseTarget) -> PulseTargetManifest:
    """Publish exactly the simulator wires with a physical-model consumer."""

    endpoints: dict[str, tuple[str, ...]] = {
        "ch00": ("SIM:C0",),
        "ch01": ("SIM:C1",),
        "ch03": ("SIM:PROBE",),
        "ch06": ("SIM:MOT_CAMERA",),
        "ch09": ("SIM:TRAP",),
        "ch11": ("SIM:READOUT_CAMERA",),
    }
    for model_axis, port_key in _VIRTUAL_MOT_COIL_PORTS.items():
        port = target.by_key[port_key]
        endpoints[port_key] = tuple(
            f"SIM:{model_axis.removeprefix('da_').upper()}{bit}"
            for bit in range(port.width)
        )
        assert port.latch_clock is not None
        endpoints[port.latch_clock] = (
            f"SIM:{model_axis.removeprefix('da_').upper()}CLK",
        )
    return pulse_target_manifest(target, endpoints)

def _bind_camera(
    broker: DeviceBroker,
    asset: InstallationAsset,
    asset_map_revision: str,
    camera: VirtualCamera,
    *,
    free_running_monitor: bool = False,
) -> tuple[BoundCapturePort, BoundCameraMonitorPort]:
    if not isinstance(free_running_monitor, bool):
        raise TypeError("free_running_monitor must be bool")
    endpoint = CameraMonitorEndpoint(
        camera,
        asset.role,
        exact_external_trigger_qualification_digest=canonical_digest(
            {
                "evidence": "target-owned deterministic in-process trigger wire",
                "adapter_type": (
                    f"{type(camera).__module__}.{type(camera).__qualname__}"
                ),
            }
        ),
        acquisition_mode=CameraAcquisitionMode.EXTERNAL_TRIGGERED,
        monitor_acquisition_mode=(
            CameraAcquisitionMode.FREE_RUNNING
            if free_running_monitor
            else CameraAcquisitionMode.EXTERNAL_TRIGGERED
        ),
    )
    attestation = bind_camera_endpoint(
        broker,
        asset,
        asset_map_revision,
        endpoint,
    )
    return (
        BoundCapturePort(attestation),
        BoundCameraMonitorPort(attestation),
    )


def _bind_sequencer(
    broker: DeviceBroker,
    asset: InstallationAsset,
    asset_map_revision: str,
    sequencer: VirtualSequencer,
    manifest: PulseTargetManifest,
) -> BoundPulsePort:
    endpoint = VirtualSequencerExecutionEndpoint(sequencer, manifest)
    binding: BoundDevice | None = None

    def current_binding() -> BoundDevice:
        if binding is None:
            raise RuntimeError("sequencer endpoint binding is not installed")
        return binding

    identity = _identity_for(asset, asset_map_revision)
    proof = broker.verify_identity(lambda: identity)
    binding = broker.bind(
        key=asset.resource_key,
        identity=proof,
        execute_command=lambda command: endpoint.execute_command(
            current_binding(),
            command,
        ),
        capability_probe=lambda: endpoint.capability_probe(current_binding()),
        close_session=lambda command: endpoint.close_session(
            current_binding(),
            command,
        ),
        interrupt_operations={SafetyOperation.SAFE_STATE: endpoint.interrupt},
    )
    return BoundPulsePort(
        broker.verify_capability(binding),
        lambda session_id, run_id, artifact_digest, point_count: (
            endpoint.observe_scan_progress(
                current_binding(),
                session_id,
                run_id,
                artifact_digest,
                point_count,
            )
        ),
        lambda session_id, run_id, artifact_digest, timeout: (
            endpoint.wait_continuous_failure(
                current_binding(),
                session_id,
                run_id,
                artifact_digest,
                timeout,
            )
        ),
    )


def _bind_rf(
    broker: DeviceBroker,
    asset: InstallationAsset,
    asset_map_revision: str,
    source: VirtualRfSource,
) -> BoundRfTablePort:
    endpoint = VirtualRfTableEndpoint(source)
    binding: BoundDevice | None = None

    def current_binding() -> BoundDevice:
        if binding is None:
            raise RuntimeError("RF endpoint binding is not installed")
        return binding

    identity = _identity_for(asset, asset_map_revision)
    proof = broker.verify_identity(lambda: identity)
    binding = broker.bind(
        key=asset.resource_key,
        identity=proof,
        execute_command=lambda command: endpoint.execute_command(
            current_binding(), command
        ),
        capability_probe=lambda: endpoint.capability_probe(current_binding()),
        close_session=lambda command: endpoint.close_session(
            current_binding(), command
        ),
        interrupt_operations={SafetyOperation.SAFE_STATE: endpoint.interrupt},
    )
    return BoundRfTablePort(broker.verify_capability(binding))


def create_virtual_installation(
    *,
    seed: int | None = 7,
    device_plan: tuple[InstallationDevicePlan, ...] | None = None,
) -> _InstallationComposition:
    """Build and publish one immutable virtual graph or fail without a partial runtime."""

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("virtual installation seed must be an integer or None")
        if seed < 0:
            raise ValueError("virtual installation seed must be non-negative")
    if device_plan is None:
        from .package import INSTALLATION_PACKAGE

        device_plan = INSTALLATION_PACKAGE.device_plan
    trap: VirtualAtomArray | None = None
    sequencer: VirtualSequencer | None = None
    rf: VirtualRfSource | None = None
    camera: VirtualCamera | None = None
    mot_camera: VirtualCamera | None = None
    devices: dict[str, object] = {}
    resources: ResourceArbiter | None = None
    broker: DeviceBroker | None = None
    try:
        target = _deployed_target()
        target_manifest = _virtual_target_manifest(target)
        frame_shape_yx, grid_shape_yx, site_centers_xy = (
            _virtual_imaging_geometry()
        )
        sequencer = VirtualSequencer(
            target,
            # Standard deployed virtual composition is one frozen config bundle.
            # Do not combine an env/cwd clock override with the compiler's shipped
            # StreamerParams geometry.
            clock_hz=default_clock_hz(DEFAULT_CONFIG_PATH),
        )
        devices["sequencer"] = sequencer
        rf = VirtualRfSource(sequencer)
        devices["rf"] = rf
        trap = VirtualAtomArray(
            frame_shape_yx=frame_shape_yx,
            grid_shape_yx=grid_shape_yx,
            site_centers_xy=site_centers_xy,
            seed=seed,
            cooling_channels=_COOLING_CHANNELS,
            probe_channels=_PROBE_CHANNELS,
            trap_channels=_TRAP_CHANNELS,
            rf=rf,
        )
        devices["trap"] = trap
        camera = VirtualCamera(
            trap,
            sequencer=sequencer,
            capture_trigger_channels=_READOUT_CAMERA_TRIGGER_CHANNELS,
        )
        devices["camera"] = camera
        mot_camera = VirtualCamera(
            VirtualMotFrameSource(
                sequencer,
                seed=None if seed is None else seed + 1,
                coil_ports=_VIRTUAL_MOT_COIL_PORTS,
            ),
            sequencer=sequencer,
            capture_trigger_channels=_MOT_CAMERA_TRIGGER_CHANNELS,
            exposure=0.05,
            free_running_live=True,
        )
        devices["mot_camera"] = mot_camera
        # The trap is a private simulator model behind the camera, not an unbound
        # public physical-device role.  It remains in the exact reverse-close graph.
        assets = InstallationAssetMap.ephemeral(
            {
                "sequencer": sequencer,
                "rf": rf,
                "camera": camera,
                "mot_camera": mot_camera,
            }
        )
        installation_id = f"installation-{assets.revision[:20]}"
        runtime_instance_id = uuid.uuid4().hex
        for role in ("sequencer", "rf", "camera", "mot_camera"):
            devices[role].ensure_open()
        sequencer.set_safe_state()
        broker = DeviceBroker()
        camera_port, readout_camera_monitor_port = _bind_camera(
            broker,
            assets.require("camera", camera),
            assets.revision,
            camera,
        )
        mot_camera_port, mot_camera_monitor_port = _bind_camera(
            broker,
            assets.require("mot_camera", mot_camera),
            assets.revision,
            mot_camera,
            free_running_monitor=True,
        )
        pulse_port = _bind_sequencer(
            broker,
            assets.require("sequencer", sequencer),
            assets.revision,
            sequencer,
            target_manifest,
        )
        rf_port = _bind_rf(
            broker,
            assets.require("rf", rf),
            assets.revision,
            rf,
        )
        readout_apparatus_facts = ReadoutApparatusFacts(
            camera_role="camera",
            sequencer_role="sequencer",
            frame_shape_yx=frame_shape_yx,
            grid_shape_yx=grid_shape_yx,
            site_centers_xy=site_centers_xy,
            trigger_channel=_READOUT_CAMERA_TRIGGER_CHANNELS[0],
        )
        catalog = _catalog(
            installation_id,
            runtime_instance_id,
            assets,
            devices,
            device_plan,
        )
        resources = ResourceArbiter()
        controller = RunController(resources)
        runtime = _InstallationRuntime(
            installation_id=installation_id,
            runtime_instance_id=runtime_instance_id,
            catalog=catalog,
            resources=resources,
            broker=broker,
            controller=controller,
            camera_ports={
                "camera": camera_port,
                "mot_camera": mot_camera_port,
            },
            camera_monitor_ports={
                "camera": readout_camera_monitor_port,
                "mot_camera": mot_camera_monitor_port,
            },
            pulse_ports={"sequencer": pulse_port},
            rf_ports={"rf": rf_port},
            raw_graph=devices,
            close_order=(
                "mot_camera",
                "camera",
                "rf",
                "sequencer",
                "trap",
            ),
        )
        return _InstallationComposition(
            runtime=runtime,
            readout_apparatus_facts=(readout_apparatus_facts,),
            camera_signal_association_authorities=(("camera", camera),),
        )
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        authority_actions = (
            None if broker is None else broker.shutdown,
            None if mot_camera is None else mot_camera.close,
            None if camera is None else camera.close,
            None if rf is None else rf.close,
            None if sequencer is None else sequencer.close,
            None if trap is None else getattr(trap, "close", None),
        )
        for action in authority_actions:
            if action is None:
                continue
            try:
                action()
            except BaseException as error:
                cleanup_errors.append(error)
        if not cleanup_errors:
            final_action = None if resources is None else resources.shutdown
            if final_action is not None:
                try:
                    final_action()
                except BaseException as error:
                    cleanup_errors.append(error)
        for error in cleanup_errors:
            try:
                primary.add_note(
                    "virtual installation startup cleanup also failed: "
                    f"{type(error).__name__}: {error}"
                )
            except BaseException:
                pass
        raise

__all__ = ["create_virtual_installation"]
