"""Independent virtual sequencer, RF, and camera device leaves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_neutral_atom.device_types import (
    CAPABILITY_CAMERA_CAPTURE,
    CAPABILITY_CAMERA_MONITOR,
    CAPABILITY_CAMERA_SIGNAL_ASSOCIATION,
    CAPABILITY_MOT_FIELD_CAPTURE,
    CAPABILITY_PULSE_EXECUTE,
    CAPABILITY_READOUT_BINDING,
    CAPABILITY_RF_TABLE,
    DeviceTypeDescriptor,
)
from zlc_neutral_atom.devices.camera.binding import bind_camera_endpoint
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.camera.contract import CameraAcquisitionMode
from zlc_neutral_atom.devices.camera.endpoint import CameraMonitorEndpoint
from zlc_neutral_atom.devices.camera.monitor import BoundCameraMonitorPort
from zlc_neutral_atom.devices.rf import BoundRfTablePort
from zlc_neutral_atom.devices.simulation.apparatus import (
    VirtualAtomArray,
    VirtualCamera,
    VirtualMotFrameSource,
    VirtualRfSource,
    VirtualSequencer,
)
from zlc_neutral_atom.devices.simulation.rf_endpoint import VirtualRfTableEndpoint
from zlc_neutral_atom.devices.simulation.sequencer_endpoint import (
    VirtualSequencerExecutionEndpoint,
)
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.installation import ReadoutInstallationBinding
from zlc_neutral_atom.installation_config import DeviceInstanceConfig
from zlc_neutral_atom.runtime.ports import BoundDevice, DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
)
from zlc_pulse import (
    PORT_DAC,
    PORT_DIGITAL,
    PulseTarget,
    PulseTargetManifest,
    load_deployed_geometry_facts,
    load_deployed_pulse_target,
    pulse_target_manifest,
)


DEFAULT_VIRTUAL_SEED = 7
VIRTUAL_SEQUENCER_CAPABILITY = "simulation.pulse_source"
VIRTUAL_RF_CAPABILITY = "simulation.rf_source"

_COOLING_CHANNELS = ("ch00", "ch01")
_PROBE_CHANNELS = ("ch03",)
_TRAP_CHANNELS = ("ch09",)
_READOUT_TRIGGER_CHANNELS = ("ch11",)
_MOT_TRIGGER_CHANNELS = ("ch06",)
_MOT_COIL_PORTS = {
    "da_x": "da_bias_x",
    "da_y": "da_bias_y",
    "da_z": "da_bias_z",
}


@dataclass(frozen=True, slots=True)
class _VirtualSequencerConnection:
    sequencer: VirtualSequencer
    manifest: PulseTargetManifest
    geometry_fingerprint: int


@dataclass(frozen=True, slots=True)
class _VirtualRfConnection:
    source: VirtualRfSource


def _imaging_geometry():
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
    target = load_deployed_pulse_target()
    required = {
        *_COOLING_CHANNELS,
        *_PROBE_CHANNELS,
        *_TRAP_CHANNELS,
        *_READOUT_TRIGGER_CHANNELS,
        *_MOT_TRIGGER_CHANNELS,
        *_MOT_COIL_PORTS.values(),
    }
    missing = tuple(sorted(required.difference(target.by_key)))
    if missing:
        raise RuntimeError(f"deployed target is missing virtual ports {missing}")
    for key in sorted(required.difference(_MOT_COIL_PORTS.values())):
        port = target.by_key[key]
        if port.kind != PORT_DIGITAL or port.lanes != (key,):
            raise RuntimeError(f"virtual wiring {key!r} is not one digital lane")
    for key in _MOT_COIL_PORTS.values():
        port = target.by_key[key]
        if port.kind != PORT_DAC or port.latch_clock is None:
            raise RuntimeError(f"virtual MOT wiring {key!r} is not a latched DAC")
    return target


def _manifest(target: PulseTarget) -> PulseTargetManifest:
    endpoints: dict[str, tuple[str, ...]] = {
        "ch00": ("SIM:C0",),
        "ch01": ("SIM:C1",),
        "ch03": ("SIM:PROBE",),
        "ch06": ("SIM:MOT_CAMERA",),
        "ch09": ("SIM:TRAP",),
        "ch11": ("SIM:READOUT_CAMERA",),
    }
    for axis, port_key in _MOT_COIL_PORTS.items():
        port = target.by_key[port_key]
        label = axis.removeprefix("da_").upper()
        endpoints[port_key] = tuple(f"SIM:{label}{bit}" for bit in range(port.width))
        assert port.latch_clock is not None
        endpoints[port.latch_clock] = (f"SIM:{label}CLK",)
    return pulse_target_manifest(target, endpoints)


def _identity(instance_id: str) -> PhysicalDeviceIdentity:
    return PhysicalDeviceIdentity(
        f"virtual:{instance_id}",
        DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
    )


def _bind_sequencer(
    broker: DeviceBroker,
    instance_id: str,
    connection: _VirtualSequencerConnection,
) -> BoundPulsePort:
    endpoint = VirtualSequencerExecutionEndpoint(
        connection.sequencer,
        connection.manifest,
        geometry_fingerprint=connection.geometry_fingerprint,
    )
    binding: BoundDevice | None = None

    def current_binding() -> BoundDevice:
        if binding is None:
            raise RuntimeError("virtual sequencer binding is not installed")
        return binding

    proof = broker.verify_identity(lambda: _identity(instance_id))
    binding = broker.bind(
        key=ResourceKey(("device", instance_id)),
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
    return BoundPulsePort(
        capability_attestation=broker.verify_capability(binding),
        editor_connection_mode="virtual",
        _scan_progress_reader=lambda session_id, run_id, digest, count: (
            endpoint.observe_scan_progress(
                current_binding(), session_id, run_id, digest, count
            )
        ),
        _continuous_failure_waiter=lambda session_id, run_id, digest, timeout: (
            endpoint.wait_continuous_failure(
                current_binding(), session_id, run_id, digest, timeout
            )
        ),
    )


def _bind_rf(
    broker: DeviceBroker,
    instance_id: str,
    source: VirtualRfSource,
) -> BoundRfTablePort:
    endpoint = VirtualRfTableEndpoint(source)
    binding: BoundDevice | None = None

    def current_binding() -> BoundDevice:
        if binding is None:
            raise RuntimeError("virtual RF binding is not installed")
        return binding

    proof = broker.verify_identity(lambda: _identity(instance_id))
    binding = broker.bind(
        key=ResourceKey(("device", instance_id)),
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


def _bind_camera(
    broker: DeviceBroker,
    instance: DeviceInstanceConfig,
    camera: VirtualCamera,
    *,
    free_running_monitor: bool,
):
    endpoint = CameraMonitorEndpoint(
        camera,
        instance.instance_id,
        exact_external_trigger_qualified=True,
        acquisition_mode=CameraAcquisitionMode.EXTERNAL_TRIGGERED,
        monitor_acquisition_mode=(
            CameraAcquisitionMode.FREE_RUNNING
            if free_running_monitor
            else CameraAcquisitionMode.EXTERNAL_TRIGGERED
        ),
    )
    attestation = bind_camera_endpoint(
        broker,
        instance_id=instance.instance_id,
        identity=_identity(instance.instance_id),
        endpoint=endpoint,
    )
    # The endpoint owns generic finite/live adapter commands.  Exact signal
    # association belongs to VirtualCamera itself because it alone observes
    # the in-process sequencer FIRE callback and frame ordinals under one lock.
    return BoundCapturePort(attestation), BoundCameraMonitorPort(attestation), camera


def _connect_sequencer(
    instance: DeviceInstanceConfig,
    _dependencies: Mapping[str, object],
    broker: object,
    required_pulse_document: object | None,
):
    if required_pulse_document is not None:
        raise ValueError("required_pulse_document is valid only for remote sequencers")
    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    geometry = load_deployed_geometry_facts()
    target = _deployed_target()
    sequencer = VirtualSequencer(target, clock_hz=geometry.clock_hz)
    try:
        sequencer.ensure_open()
        sequencer.set_safe_state()
        connection = _VirtualSequencerConnection(
            sequencer,
            _manifest(target),
            geometry.geometry_fingerprint,
        )
        pulse_port = _bind_sequencer(broker, instance.instance_id, connection)
    except BaseException as primary:
        try:
            sequencer.close()
        except BaseException as error:
            primary.add_note(f"virtual sequencer close also failed: {error}")
        raise
    return {
        CAPABILITY_PULSE_EXECUTE: pulse_port,
        VIRTUAL_SEQUENCER_CAPABILITY: connection,
    }, sequencer.close


def _connect_rf(
    instance: DeviceInstanceConfig,
    dependencies: Mapping[str, object],
    broker: object,
    _required_pulse_document: object | None,
):
    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    connection = dependencies["sequencer_ref"]
    if not isinstance(connection, _VirtualSequencerConnection):
        raise TypeError("virtual RF requires a virtual sequencer")
    source = VirtualRfSource(connection.sequencer)
    try:
        source.ensure_open()
        port = _bind_rf(broker, instance.instance_id, source)
    except BaseException as primary:
        try:
            source.close()
        except BaseException as error:
            primary.add_note(f"virtual RF close also failed: {error}")
        raise
    return {
        CAPABILITY_RF_TABLE: port,
        VIRTUAL_RF_CAPABILITY: _VirtualRfConnection(source),
    }, source.close


def _connect_readout_camera(
    instance: DeviceInstanceConfig,
    dependencies: Mapping[str, object],
    broker: object,
    _required_pulse_document: object | None,
):
    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    seq = dependencies["sequencer_ref"]
    rf = dependencies["rf_ref"]
    if not isinstance(seq, _VirtualSequencerConnection) or not isinstance(
        rf,
        _VirtualRfConnection,
    ):
        raise TypeError("virtual readout camera dependencies have wrong types")
    values = _READOUT_SCHEMA.freeze(instance.parameters)
    frame_shape, grid_shape, centers = _imaging_geometry()
    trap = VirtualAtomArray(
        frame_shape_yx=frame_shape,
        grid_shape_yx=grid_shape,
        site_centers_xy=centers,
        seed=values["seed"],
        cooling_channels=_COOLING_CHANNELS,
        probe_channels=_PROBE_CHANNELS,
        trap_channels=_TRAP_CHANNELS,
        rf=rf.source,
    )
    camera = VirtualCamera(
        trap,
        sequencer=seq.sequencer,
        capture_trigger_channels=_READOUT_TRIGGER_CHANNELS,
    )
    try:
        camera.ensure_open()
        capture, monitor, authority = _bind_camera(
            broker,
            instance,
            camera,
            free_running_monitor=False,
        )
        readout_binding = ReadoutInstallationBinding(
            camera_instance_id=instance.instance_id,
            sequencer_instance_id=instance.parameters["sequencer_ref"],
            trigger_channel=_READOUT_TRIGGER_CHANNELS[0],
        )
    except BaseException as primary:
        try:
            camera.close()
        except BaseException as error:
            primary.add_note(f"virtual camera close also failed: {error}")
        raise
    return {
        CAPABILITY_CAMERA_CAPTURE: capture,
        CAPABILITY_CAMERA_MONITOR: monitor,
        CAPABILITY_CAMERA_SIGNAL_ASSOCIATION: authority,
        CAPABILITY_READOUT_BINDING: readout_binding,
    }, camera.close


def _connect_mot_camera(
    instance: DeviceInstanceConfig,
    dependencies: Mapping[str, object],
    broker: object,
    _required_pulse_document: object | None,
):
    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    seq = dependencies["sequencer_ref"]
    if not isinstance(seq, _VirtualSequencerConnection):
        raise TypeError("virtual MOT camera requires a virtual sequencer")
    values = _MOT_CAMERA_SCHEMA.freeze(instance.parameters)
    seed = values["seed"]
    camera = VirtualCamera(
        VirtualMotFrameSource(
            seq.sequencer,
            seed=None if seed is None else seed + 1,
            coil_ports=_MOT_COIL_PORTS,
        ),
        sequencer=seq.sequencer,
        capture_trigger_channels=_MOT_TRIGGER_CHANNELS,
        exposure=0.05,
        free_running_live=True,
    )
    try:
        camera.ensure_open()
        capture, monitor, _authority = _bind_camera(
            broker,
            instance,
            camera,
            free_running_monitor=True,
        )
    except BaseException as primary:
        try:
            camera.close()
        except BaseException as error:
            primary.add_note(f"virtual MOT camera close also failed: {error}")
        raise
    return {
        CAPABILITY_CAMERA_CAPTURE: capture,
        CAPABILITY_CAMERA_MONITOR: monitor,
        CAPABILITY_MOT_FIELD_CAPTURE: capture,
    }, camera.close


_EMPTY_SCHEMA = AuthoringSchema(())
_RF_SCHEMA = AuthoringSchema(
    (AuthoringField("sequencer_ref", "text", "Sequencer", "sequencer", True),)
)
_READOUT_SCHEMA = AuthoringSchema(
    (
        AuthoringField("sequencer_ref", "text", "Sequencer", "sequencer", True),
        AuthoringField("rf_ref", "text", "RF source", "rf", True),
        AuthoringField(
            "seed",
            "int",
            "Random seed",
            DEFAULT_VIRTUAL_SEED,
            False,
            minimum=0,
            allow_blank=True,
        ),
    )
)
_MOT_CAMERA_SCHEMA = AuthoringSchema(
    (
        AuthoringField("sequencer_ref", "text", "Sequencer", "sequencer", True),
        AuthoringField(
            "seed",
            "int",
            "Random seed",
            DEFAULT_VIRTUAL_SEED,
            False,
            minimum=0,
            allow_blank=True,
        ),
    )
)


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        "sequencer.virtual",
        "sequencer",
        "Virtual sequencer",
        _EMPTY_SCHEMA,
        (CAPABILITY_PULSE_EXECUTE, VIRTUAL_SEQUENCER_CAPABILITY),
        (),
        _connect_sequencer,
    ),
    DeviceTypeDescriptor(
        "rf.virtual",
        "rf",
        "Virtual RF source",
        _RF_SCHEMA,
        (CAPABILITY_RF_TABLE, VIRTUAL_RF_CAPABILITY),
        (("sequencer_ref", VIRTUAL_SEQUENCER_CAPABILITY),),
        _connect_rf,
    ),
    DeviceTypeDescriptor(
        "camera.virtual_readout",
        "camera",
        "Virtual readout camera",
        _READOUT_SCHEMA,
        (
            CAPABILITY_CAMERA_CAPTURE,
            CAPABILITY_CAMERA_MONITOR,
            CAPABILITY_CAMERA_SIGNAL_ASSOCIATION,
            CAPABILITY_READOUT_BINDING,
        ),
        (
            ("sequencer_ref", VIRTUAL_SEQUENCER_CAPABILITY),
            ("rf_ref", VIRTUAL_RF_CAPABILITY),
        ),
        _connect_readout_camera,
    ),
    DeviceTypeDescriptor(
        "camera.virtual_mot",
        "camera",
        "Virtual MOT camera",
        _MOT_CAMERA_SCHEMA,
        (
            CAPABILITY_CAMERA_CAPTURE,
            CAPABILITY_CAMERA_MONITOR,
            CAPABILITY_MOT_FIELD_CAPTURE,
        ),
        (("sequencer_ref", VIRTUAL_SEQUENCER_CAPABILITY),),
        _connect_mot_camera,
    ),
)


__all__ = ["DEVICE_TYPES", "VIRTUAL_RF_CAPABILITY", "VIRTUAL_SEQUENCER_CAPABILITY"]
