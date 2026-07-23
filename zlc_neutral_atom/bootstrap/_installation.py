"""One owned neutral-atom installation composition."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Mapping

from fpga.pulse_streamer.host.image import (
    DEFAULT_CONFIG_PATH,
    StreamerParams,
    build_fingerprint,
    default_clock_hz,
)
from zlc_neutral_atom.acquisition.camera import CameraAcquisitionMode
from zlc_neutral_atom.installation import (
    DeviceCatalogView,
    DeviceInfo,
    DeviceRef,
)
from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
    RemotePulseInstallationConfig,
    VirtualInstallationConfig,
)
from zlc_neutral_atom.readout.calibration import GridOrder
from zlc_neutral_atom.readout.contracts import (
    ReadoutBindingKey,
    camera_roi_local_spatial_identity,
)
from zlc_neutral_atom.readout.sitemap import (
    ReadoutGridGeometry,
    SitemapAcquisitionProfile,
    load_packaged_sitemap_pulse,
)
from zlc_neutral_atom.runtime.capture import BoundCapturePort
from zlc_neutral_atom.runtime.monitor import BoundCameraMonitorPort
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    DeviceBroker,
    SafetyOperation,
)
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceArbiter,
    ResourceKey,
)
from zlc_neutral_atom.runtime.run import RunController, RunHandle, RunPlan
from zlc_neutral_atom.timing.pulse import BoundPulsePort
from zlc_pulse import (
    PORT_DAC,
    PORT_DIGITAL,
    PulseDocument,
    PulseTarget,
    PulseTargetManifest,
    RemotePulseExecutionClient,
    bind_pulse_document_target,
    load_deployed_pulse_target,
    pulse_target_manifest,
    validate_pulse_document_clock_grid,
)
from zlc_storage import canonical_digest, normalized_text

from ._asset_map import InstallationAsset, InstallationAssetMap, adapter_kind
from ._camera_endpoint import CameraCaptureEndpoint, CameraMonitorEndpoint
from ._sequencer_endpoint import (
    RemotePulseExecutionEndpoint,
    VirtualSequencerExecutionEndpoint,
)
from ._virtual_hardware import (
    VirtualAtomArray,
    VirtualCamera,
    VirtualMotFrameSource,
    VirtualMonitorCamera,
    VirtualSequencer,
)


# Installation wiring, not a friendly simulator alias.  These are the physical
# lanes used by the checked-in deployed PulseTarget and by the real apparatus.
_COOLING_CHANNELS = ("ch00", "ch01")
_PROBE_CHANNELS = ("ch03",)
_TRAP_CHANNELS = ("ch09",)
_CAMERA_TRIGGER_CHANNELS = ("ch11",)
_VIRTUAL_MOT_COIL_PORTS = {
    "da_x": "da_bias_x",
    "da_y": "da_bias_y",
    "da_z": "da_bias_z",
}


def _virtual_readout_geometry() -> ReadoutGridGeometry:
    """The one installation source for simulator and calibration site geometry."""

    grid_shape_yx = (5, 7)
    frame_shape_yx = (96, 128)
    spacing_px = 9.0
    rows, columns = grid_shape_yx
    height, width = frame_shape_yx
    origin_x = (width - (columns - 1) * spacing_px) / 2.0
    origin_y = (height - (rows - 1) * spacing_px) / 2.0
    y_axis_id, x_axis_id, coordinate_frame = camera_roi_local_spatial_identity(
        "camera"
    )
    return ReadoutGridGeometry(
        frame_shape_yx,
        y_axis_id,
        x_axis_id,
        coordinate_frame,
        grid_shape_yx,
        GridOrder.ROW_MAJOR,
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
        *_CAMERA_TRIGGER_CHANNELS,
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
        "ch09": ("SIM:TRAP",),
        "ch11": ("SIM:CAM",),
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


def _identity_for(
    asset: InstallationAsset,
    asset_map_revision: str,
) -> PhysicalDeviceIdentity:
    if (
        asset.evidence_kind
        is not DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT
    ):
        raise ValueError("this asset requires installation-asserted identity")
    return PhysicalDeviceIdentity(
        stable_device_identity=asset.expected_identity,
        evidence_kind=asset.evidence_kind,
        evidence_digest=canonical_digest(
            {
                "asset_id": asset.asset_id,
                "role": asset.role,
                "adapter_kind": asset.adapter_kind,
                "expected_identity": asset.expected_identity,
            }
        ),
        asset_map_revision=asset_map_revision,
    )


def _bind_camera(
    broker: DeviceBroker,
    asset: InstallationAsset,
    asset_map_revision: str,
    camera: VirtualCamera,
) -> tuple[BoundCapturePort, BoundCameraMonitorPort]:
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
    )
    attestation = _bind_camera_endpoint(
        broker,
        asset,
        asset_map_revision,
        endpoint,
    )
    return (
        BoundCapturePort(attestation),
        BoundCameraMonitorPort(attestation),
    )


def _bind_camera_endpoint(
    broker: DeviceBroker,
    asset: InstallationAsset,
    asset_map_revision: str,
    endpoint: CameraCaptureEndpoint | CameraMonitorEndpoint,
):
    """Bind the one shared camera command/cleanup/safe-state surface."""

    binding: BoundDevice | None = None

    def current_binding() -> BoundDevice:
        if binding is None:
            raise RuntimeError("camera endpoint binding is not installed")
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
        interrupt_operations={SafetyOperation.DISARM: endpoint.interrupt},
    )
    return broker.verify_capability(binding)


def _bind_monitor_camera(
    broker: DeviceBroker,
    asset: InstallationAsset,
    asset_map_revision: str,
    camera: VirtualMonitorCamera,
) -> BoundCameraMonitorPort:
    endpoint = CameraMonitorEndpoint(camera, asset.role)
    return BoundCameraMonitorPort(
        _bind_camera_endpoint(
            broker,
            asset,
            asset_map_revision,
            endpoint,
        )
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


def _bind_remote_sequencer(
    broker: DeviceBroker,
    asset: InstallationAsset,
    asset_map_revision: str,
    client: RemotePulseExecutionClient,
    *,
    endpoint_label: str,
    max_blocking_call_seconds: float | None,
) -> BoundPulsePort:
    """Bind the current remote protocol behind the same typed pulse Port.

    The GUI and notebook never receive ``client`` or the endpoint.  They submit a
    declarative PulseRunRequest through the ordinary RunController, so remote use
    cannot grow a second prepare/fire/safe authority.
    """

    endpoint = RemotePulseExecutionEndpoint(
        client,
        endpoint_label=endpoint_label,
        max_blocking_call_seconds=max_blocking_call_seconds,
    )
    binding: BoundDevice | None = None

    def current_binding() -> BoundDevice:
        if binding is None:
            raise RuntimeError("remote sequencer endpoint binding is not installed")
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


class _InstallationRuntime:
    """Process-lifetime owner of one immutable installation graph."""

    __slots__ = (
        "_lock",
        "_shutdown_lock",
        "_state",
        "_installation_id",
        "_runtime_instance_id",
        "_catalog",
        "_resources",
        "_broker",
        "_controller",
        "_camera_ports",
        "_camera_monitor_ports",
        "_pulse_ports",
        "_sitemap_profiles",
        "_raw_graph",
        "_close_order",
        "_closed_roles",
        "_broker_closed",
        "_resources_closed",
        "_shutdown_diagnostics",
    )

    def __init__(
        self,
        *,
        installation_id: str,
        runtime_instance_id: str,
        catalog: DeviceCatalogView,
        resources: ResourceArbiter,
        broker: DeviceBroker,
        controller: RunController,
        camera_ports: Mapping[str, BoundCapturePort],
        camera_monitor_ports: Mapping[str, BoundCameraMonitorPort],
        pulse_ports: Mapping[str, BoundPulsePort],
        sitemap_profiles: Mapping[str, SitemapAcquisitionProfile],
        raw_graph: Mapping[str, object],
        close_order: tuple[str, ...],
    ) -> None:
        self._lock = threading.RLock()
        self._shutdown_lock = threading.Lock()
        self._state = "RUNNING"
        self._installation_id = installation_id
        self._runtime_instance_id = runtime_instance_id
        self._catalog = catalog
        self._resources = resources
        self._broker = broker
        self._controller = controller
        camera_ports = dict(camera_ports)
        camera_monitor_ports = dict(camera_monitor_ports)
        pulse_ports = dict(pulse_ports)
        sitemap_profiles = dict(sitemap_profiles)
        raw_graph = dict(raw_graph)
        close_order = tuple(close_order)
        if len(close_order) != len(set(close_order)):
            raise ValueError("installation close order contains duplicate roles")
        if set(close_order) != set(raw_graph):
            raise ValueError(
                "installation close order must cover the raw graph exactly"
            )
        public_roles = set(camera_ports) | set(camera_monitor_ports) | set(pulse_ports)
        if not public_roles.issubset(raw_graph):
            raise ValueError("installation ports reference roles outside the raw graph")
        if not public_roles.issubset(set(catalog)):
            raise ValueError("installation ports reference roles outside the catalog")
        if any(catalog.require(role).domain != "camera" for role in camera_ports):
            raise ValueError("camera port roles differ from catalog domains")
        if any(
            catalog.require(role).domain != "camera" for role in camera_monitor_ports
        ):
            raise ValueError("camera monitor port roles differ from catalog domains")
        if any(
            catalog.require(role).domain != "sequencer" for role in pulse_ports
        ):
            raise ValueError("pulse port roles differ from catalog domains")
        if not set(sitemap_profiles).issubset(camera_ports):
            raise ValueError("sitemap profiles reference roles without camera ports")
        for role, profile in sitemap_profiles.items():
            if not isinstance(profile, SitemapAcquisitionProfile):
                raise TypeError("sitemap profile mapping has the wrong value type")
            if profile.readout_binding != ReadoutBindingKey(role):
                raise ValueError("sitemap profile binding differs from its camera role")
            if profile.sequencer_role not in pulse_ports:
                raise ValueError(
                    "sitemap profile references a role without a sequencer port"
                )
            if (
                profile.camera_facts
                != camera_ports[role].capability.camera_physical_facts
            ):
                raise ValueError(
                    "sitemap profile camera facts differ from the bound capability"
                )
            live_target = pulse_ports[profile.sequencer_role].capability.target
            if profile.pulse_document.target is not live_target:
                raise ValueError("sitemap pulse is not bound to the exact live target")
        self._camera_ports = camera_ports
        self._camera_monitor_ports = camera_monitor_ports
        self._pulse_ports = pulse_ports
        self._sitemap_profiles = sitemap_profiles
        self._raw_graph = raw_graph
        self._close_order = close_order
        self._closed_roles: set[str] = set()
        self._broker_closed = False
        self._resources_closed = False
        self._shutdown_diagnostics: tuple[str, ...] = ()

    @property
    def device_catalog(self) -> DeviceCatalogView:
        return self._catalog

    @property
    def runtime_instance_id(self) -> str:
        return self._runtime_instance_id

    @property
    def shutdown_diagnostics(self) -> tuple[str, ...]:
        """Detached failures from the most recent close attempt."""

        with self._lock:
            return self._shutdown_diagnostics

    def camera_port(self, reference: DeviceRef) -> BoundCapturePort:
        with self._lock:
            if self._state != "RUNNING":
                raise RuntimeError("installation runtime is not accepting operations")
            self._require_current_reference(reference, "camera")
            try:
                return self._camera_ports[reference.role]
            except KeyError as exc:
                raise ValueError(f"role {reference.role!r} is not a camera") from exc

    def camera_monitor_port(self, reference: DeviceRef) -> BoundCameraMonitorPort:
        with self._lock:
            if self._state != "RUNNING":
                raise RuntimeError("installation runtime is not accepting operations")
            self._require_current_reference(reference, "camera")
            try:
                return self._camera_monitor_ports[reference.role]
            except KeyError as exc:
                raise ValueError(
                    f"camera role {reference.role!r} is not free-running monitor capable"
                ) from exc

    def pulse_port(self, reference: DeviceRef) -> BoundPulsePort:
        with self._lock:
            if self._state != "RUNNING":
                raise RuntimeError("installation runtime is not accepting operations")
            self._require_current_reference(reference, "sequencer")
            try:
                return self._pulse_ports[reference.role]
            except KeyError as exc:
                raise ValueError(f"role {reference.role!r} is not a sequencer") from exc

    def sitemap_profile(self, reference: DeviceRef) -> SitemapAcquisitionProfile:
        """Return one immutable domain descriptor, never the private camera/trap."""

        with self._lock:
            if self._state != "RUNNING":
                raise RuntimeError("installation runtime is not accepting operations")
            self._require_current_reference(reference, "camera")
            try:
                return self._sitemap_profiles[reference.role]
            except KeyError as exc:
                raise ValueError(
                    f"camera role {reference.role!r} has no sitemap profile"
                ) from exc

    def start(self, plan: RunPlan) -> RunHandle:
        with self._lock:
            if self._state != "RUNNING":
                raise RuntimeError("installation runtime is not accepting Runs")
            return self._controller.start(plan)

    def wait(self, handle: RunHandle):
        """Wait through the controller without holding the runtime lifecycle lock."""

        with self._lock:
            controller = self._controller
        return controller.wait(handle)

    def _require_current_reference(self, reference: DeviceRef, domain: str) -> None:
        if not isinstance(reference, DeviceRef):
            raise TypeError("device reference must be DeviceRef")
        if (
            reference.installation_id != self._installation_id
            or reference.runtime_instance_id != self._runtime_instance_id
        ):
            raise RuntimeError("device reference belongs to another runtime instance")
        info = self._catalog.require(reference.role)
        if info.domain != domain:
            raise ValueError(
                f"device role {reference.role!r} is {info.domain!r}, not {domain!r}"
            )

    def shutdown(self, *, timeout: float = 2.0) -> bool:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("shutdown timeout must be a number")
        timeout = float(timeout)
        if timeout < 0:
            raise ValueError("shutdown timeout must be non-negative")
        deadline = time.monotonic() + timeout
        if not self._shutdown_lock.acquire(timeout=max(0.0, timeout)):
            return False
        try:
            return self._shutdown_owned(deadline)
        finally:
            self._shutdown_lock.release()

    def _shutdown_owned(self, deadline: float) -> bool:
        with self._lock:
            if self._state == "CLOSED":
                return True
            self._state = "CLOSING"
        # Never hold the lifecycle lock across controller joins, adapter/SDK
        # close, or broker invalidation.
        if not self._controller.shutdown(
            max(0.0, deadline - time.monotonic())
        ):
            with self._lock:
                self._shutdown_diagnostics = (
                    "active Run did not terminate before the shutdown deadline",
                )
            return False
        failures: list[str] = []
        if not self._broker_closed:
            try:
                self._broker.shutdown()
            except BaseException as error:
                failures.append(
                    f"device broker: {type(error).__name__}: {error}"
                )
            else:
                self._broker_closed = True
        for role in self._close_order:
            if role in self._closed_roles:
                continue
            close = getattr(self._raw_graph[role], "close", None)
            try:
                if callable(close):
                    close()
            except BaseException as error:
                failures.append(
                    f"{role}: {type(error).__name__}: {error}"
                )
            else:
                self._closed_roles.add(role)
        if failures:
            with self._lock:
                self._shutdown_diagnostics = tuple(failures)
            return False
        # Release process-local ownership only after every raw close acknowledges.
        if not self._resources_closed:
            try:
                self._resources.shutdown()
            except BaseException as error:
                failures.append(
                    f"resource arbiter: {type(error).__name__}: {error}"
                )
            else:
                self._resources_closed = True
        if failures:
            with self._lock:
                self._shutdown_diagnostics = tuple(failures)
            return False
        with self._lock:
            self._state = "CLOSED"
            self._shutdown_diagnostics = ()
            self._raw_graph.clear()
            self._camera_ports.clear()
            self._camera_monitor_ports.clear()
            self._pulse_ports.clear()
            self._sitemap_profiles.clear()
            return True


def create_installation(
    document: InstallationConfigDocument,
    *,
    required_pulse_document: PulseDocument | None = None,
) -> _InstallationRuntime:
    """Compose exactly one of the current typed installation variants."""

    if not isinstance(document, InstallationConfigDocument):
        raise TypeError("document must be InstallationConfigDocument")
    config = document.config
    if isinstance(config, VirtualInstallationConfig):
        if required_pulse_document is not None:
            raise ValueError(
                "required_pulse_document is valid only for remote_pulse"
            )
        return create_virtual_installation(seed=config.seed)
    if isinstance(config, RemotePulseInstallationConfig):
        return create_remote_pulse_installation(
            host=config.host,
            port=config.port,
            transport_timeout_seconds=config.transport_timeout_seconds,
            required_pulse_document=required_pulse_document,
        )
    raise TypeError("document contains an unknown installation config")


def _catalog(
    installation_id: str,
    runtime_instance_id: str,
    assets: InstallationAssetMap,
    devices: Mapping[str, object],
) -> DeviceCatalogView:
    domains = {
        "camera": "camera",
        "mot_camera": "camera",
        "monitor_camera": "camera",
        "sequencer": "sequencer",
        "trap": "trap",
    }
    return DeviceCatalogView(
        installation_id,
        runtime_instance_id,
        0,
        tuple(
            DeviceInfo(
                DeviceRef(installation_id, runtime_instance_id, asset.role),
                domains[asset.role],
                asset.adapter_kind,
                str(asset.resource_key),
            )
            for asset in assets.assets
            if asset.role in devices
            and asset.role in domains
            and asset.role != "trap"
        ),
    )


def create_virtual_installation(
    *,
    seed: int | None = 7,
) -> _InstallationRuntime:
    """Build and publish one immutable virtual graph or fail without a partial runtime."""

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("virtual installation seed must be an integer or None")
        if seed < 0:
            raise ValueError("virtual installation seed must be non-negative")
    trap: VirtualAtomArray | None = None
    sequencer: VirtualSequencer | None = None
    camera: VirtualCamera | None = None
    mot_camera: VirtualCamera | None = None
    monitor_camera: VirtualMonitorCamera | None = None
    devices: dict[str, object] = {}
    resources: ResourceArbiter | None = None
    broker: DeviceBroker | None = None
    try:
        target = _deployed_target()
        target_manifest = _virtual_target_manifest(target)
        readout_geometry = _virtual_readout_geometry()
        trap = VirtualAtomArray(
            geometry=readout_geometry,
            seed=seed,
            cooling_channels=_COOLING_CHANNELS,
            probe_channels=_PROBE_CHANNELS,
            trap_channels=_TRAP_CHANNELS,
        )
        devices["trap"] = trap
        sequencer = VirtualSequencer(
            target,
            # Standard deployed virtual composition is one frozen config bundle.
            # Do not combine an env/cwd clock override with the compiler's shipped
            # StreamerParams geometry.
            clock_hz=default_clock_hz(DEFAULT_CONFIG_PATH),
        )
        devices["sequencer"] = sequencer
        camera = VirtualCamera(
            trap,
            sequencer=sequencer,
            capture_trigger_channels=_CAMERA_TRIGGER_CHANNELS,
        )
        devices["camera"] = camera
        mot_camera = VirtualCamera(
            VirtualMotFrameSource(
                sequencer,
                seed=None if seed is None else seed + 1,
                coil_ports=_VIRTUAL_MOT_COIL_PORTS,
            ),
            sequencer=sequencer,
            capture_trigger_channels=_CAMERA_TRIGGER_CHANNELS,
            exposure=0.05,
        )
        devices["mot_camera"] = mot_camera
        monitor_camera = VirtualMonitorCamera(
            sequencer,
            coil_ports=_VIRTUAL_MOT_COIL_PORTS,
        )
        devices["monitor_camera"] = monitor_camera
        # The trap is a private simulator model behind the camera, not an unbound
        # public physical-device role.  It remains in the exact reverse-close graph.
        assets = InstallationAssetMap.ephemeral(
            {
                "sequencer": sequencer,
                "camera": camera,
                "mot_camera": mot_camera,
                "monitor_camera": monitor_camera,
            }
        )
        installation_id = f"installation-{assets.revision[:20]}"
        runtime_instance_id = uuid.uuid4().hex
        for role in ("sequencer", "camera", "mot_camera", "monitor_camera"):
            devices[role].ensure_open()
        sequencer.set_safe_state()
        broker = DeviceBroker()
        camera_port, readout_camera_monitor_port = _bind_camera(
            broker,
            assets.require("camera", camera),
            assets.revision,
            camera,
        )
        camera_monitor_port = _bind_monitor_camera(
            broker,
            assets.require("monitor_camera", monitor_camera),
            assets.revision,
            monitor_camera,
        )
        mot_camera_port, mot_camera_monitor_port = _bind_camera(
            broker,
            assets.require("mot_camera", mot_camera),
            assets.revision,
            mot_camera,
        )
        pulse_port = _bind_sequencer(
            broker,
            assets.require("sequencer", sequencer),
            assets.revision,
            sequencer,
            target_manifest,
        )
        sitemap_profile = SitemapAcquisitionProfile(
            readout_binding=ReadoutBindingKey("camera"),
            sequencer_role="sequencer",
            camera_facts=camera_port.capability.camera_physical_facts,
            geometry=readout_geometry,
            maximum_site_residual_px=2.0,
            pulse_document=bind_pulse_document_target(
                load_packaged_sitemap_pulse(),
                pulse_port.capability.target,
            ),
            trigger_channel=_CAMERA_TRIGGER_CHANNELS[0],
        )
        catalog = _catalog(
            installation_id,
            runtime_instance_id,
            assets,
            devices,
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
                "monitor_camera": camera_monitor_port,
            },
            pulse_ports={"sequencer": pulse_port},
            sitemap_profiles={"camera": sitemap_profile},
            raw_graph=devices,
            close_order=(
                "monitor_camera",
                "mot_camera",
                "camera",
                "sequencer",
                "trap",
            ),
        )
        return runtime
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        authority_actions = (
            None if broker is None else broker.shutdown,
            None if monitor_camera is None else monitor_camera.close,
            None if mot_camera is None else mot_camera.close,
            None if camera is None else camera.close,
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


def create_remote_pulse_installation(
    *,
    host: str,
    port: int = 18861,
    transport_timeout_seconds: float = 120.0,
    max_blocking_call_seconds: float | None = None,
    required_pulse_document: PulseDocument | None = None,
) -> _InstallationRuntime:
    """Compose one sequencer-only installation over the current pulse RPC.

    Network reachability is proven before publishing the owned installation.
    A typo or an unavailable server therefore remains retryable.  Once the remote
    generation has entered SAFE and is bound, all later failures are treated like
    any other partial hardware startup and fail closed.
    """

    endpoint_host = normalized_text(host, "remote pulse host")
    if isinstance(port, bool) or not isinstance(port, int):
        raise TypeError("remote pulse port must be an integer")
    if not 1 <= port <= 65535:
        raise ValueError("remote pulse port must be between 1 and 65535")
    if isinstance(transport_timeout_seconds, bool) or not isinstance(
        transport_timeout_seconds,
        (int, float),
    ):
        raise TypeError("transport_timeout_seconds must be a number")
    transport_timeout = float(transport_timeout_seconds)
    if not math.isfinite(transport_timeout) or transport_timeout <= 0:
        raise ValueError("transport_timeout_seconds must be finite and positive")
    if max_blocking_call_seconds is None:
        blocking_limit = None
    else:
        if isinstance(max_blocking_call_seconds, bool) or not isinstance(
            max_blocking_call_seconds,
            (int, float),
        ):
            raise TypeError("max_blocking_call_seconds must be a number or None")
        blocking_limit = float(max_blocking_call_seconds)
        if not math.isfinite(blocking_limit) or blocking_limit <= 0:
            raise ValueError("max_blocking_call_seconds must be finite and positive")
        if blocking_limit >= transport_timeout:
            raise ValueError(
                "max_blocking_call_seconds must be shorter than the transport timeout"
            )
    if required_pulse_document is not None and not isinstance(
        required_pulse_document,
        PulseDocument,
    ):
        raise TypeError("required_pulse_document must be PulseDocument or None")
    client = RemotePulseExecutionClient.connect(
        endpoint_host,
        port,
        transport_timeout_seconds=transport_timeout,
    )
    try:
        snapshot = client.snapshot()
        host_geometry = build_fingerprint(StreamerParams())
        if snapshot.geometry_fingerprint != host_geometry:
            raise ValueError(
                "remote pulse geometry differs from the current host compiler "
                f"geometry: remote=0x{snapshot.geometry_fingerprint:08x}, "
                f"host=0x{host_geometry:08x}"
            )
        if required_pulse_document is not None:
            bind_pulse_document_target(
                required_pulse_document,
                snapshot.target,
            )
            validate_pulse_document_clock_grid(
                required_pulse_document,
                snapshot.clock_hz,
            )
    except BaseException as primary:
        try:
            client.close()
        except BaseException as close_error:
            try:
                primary.add_note(
                    "close error: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            except BaseException:
                pass
        raise
    endpoint_label = f"{endpoint_host}:{port}"
    devices: dict[str, object] = {"sequencer": client}
    resources: ResourceArbiter | None = None
    broker: DeviceBroker | None = None
    try:
        safe_timeout = (
            client.transport_timeout_seconds * 0.9
            if max_blocking_call_seconds is None
            else blocking_limit
        )
        client.safe_state(timeout=safe_timeout)
        endpoint_identity = canonical_digest(
            {
                "protocol": "zlc.current-pulse-rpc",
                "host": endpoint_host,
                "port": port,
            }
        )
        assets = InstallationAssetMap(
            (
                InstallationAsset(
                    asset_id="remote-sequencer",
                    role="sequencer",
                    resource_key=ResourceKey.parse("device/sequencer"),
                    adapter_kind=adapter_kind(client),
                    evidence_kind=(
                        DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT
                    ),
                    expected_identity=f"remote-pulse-endpoint:{endpoint_identity}",
                ),
            )
        )
        installation_id = f"installation-{assets.revision[:20]}"
        runtime_instance_id = uuid.uuid4().hex
        broker = DeviceBroker()
        pulse_port = _bind_remote_sequencer(
            broker,
            assets.require("sequencer", client),
            assets.revision,
            client,
            endpoint_label=endpoint_label,
            max_blocking_call_seconds=blocking_limit,
        )
        catalog = _catalog(
            installation_id,
            runtime_instance_id,
            assets,
            devices,
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
            camera_ports={},
            camera_monitor_ports={},
            pulse_ports={"sequencer": pulse_port},
            sitemap_profiles={},
            raw_graph=devices,
            close_order=("sequencer",),
        )
        return runtime
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        for action in (
            None if broker is None else broker.shutdown,
            client.close,
        ):
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
                    "remote pulse installation startup cleanup also failed: "
                    f"{type(error).__name__}: {error}"
                )
            except BaseException:
                pass
        raise


__all__ = [
    "create_installation",
    "create_remote_pulse_installation",
    "create_virtual_installation",
]
