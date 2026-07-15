"""One process-lifetime neutral-atom installation composition authority."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from fpga.pulse_streamer.host.image import DEFAULT_CONFIG_PATH, default_clock_hz
from zlc_neutral_atom.installation import DeviceCatalogView, DeviceInfo, DeviceRef
from zlc_neutral_atom.runtime.capture import BoundCapturePort
from zlc_neutral_atom.runtime.ports import BoundDevice, DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceArbiter,
)
from zlc_neutral_atom.runtime.run import RunController, RunHandle, RunPlan
from zlc_neutral_atom.runtime.safety_journal import PersistentSafetyJournal
from zlc_neutral_atom.timing.pulse import BoundPulsePort
from zlc_pulse import PORT_DIGITAL, PulseTarget, load_deployed_pulse_target
from zlc_storage import canonical_digest, durable_mkdir

from ._asset_map import InstallationAsset, InstallationAssetMap
from ._camera_endpoint import CameraCaptureEndpoint
from ._sequencer_endpoint import VirtualSequencerExecutionEndpoint
from ._virtual_hardware import (
    VirtualAtomArray,
    VirtualCamera,
    VirtualSequencer,
)


# Installation wiring, not a friendly simulator alias.  These are the physical
# lanes used by the checked-in deployed PulseTarget and by the real apparatus.
_COOLING_CHANNELS = ("ch00", "ch01")
_PROBE_CHANNELS = ("ch03",)
_TRAP_CHANNELS = ("ch09",)
_CAMERA_TRIGGER_CHANNELS = ("ch11",)


@dataclass(frozen=True)
class _FailedVirtualStartupAuthority:
    """Strongly retain an unsafe partial startup until process replacement.

    This value has one deliberately narrow purpose: without it, an exceptional
    virtual adapter close could let Python garbage collection release the last
    journal/graph owner while startup is reporting failure.  It is never exposed
    as a recovery API; a later composition attempt fails closed and the process
    must be replaced.
    """

    raw_graph: Mapping[str, object]
    broker: DeviceBroker | None
    resources: ResourceArbiter | None
    journal: PersistentSafetyJournal | None


_FAILED_STARTUP_LOCK = threading.Lock()
_FAILED_STARTUPS: list[_FailedVirtualStartupAuthority] = []
_COMPOSITION_LOCK = threading.Lock()
_COMPOSITION_CLAIMED = False
_PROCESS_RUNTIME: _VirtualInstallationRuntime | None = None


def _require_no_failed_startup() -> None:
    with _FAILED_STARTUP_LOCK:
        if _FAILED_STARTUPS:
            raise RuntimeError(
                "a previous virtual installation startup failed to close cleanly; "
                "replace this process before composing another installation"
            )


def _claim_process_composition() -> None:
    global _COMPOSITION_CLAIMED
    with _COMPOSITION_LOCK:
        if _COMPOSITION_CLAIMED:
            raise RuntimeError(
                "this process already created an installation runtime; "
                "replace the process instead of rebuilding or hot-swapping it"
            )
        _COMPOSITION_CLAIMED = True


def _retain_process_runtime(runtime: "_VirtualInstallationRuntime") -> None:
    global _PROCESS_RUNTIME
    with _COMPOSITION_LOCK:
        if _PROCESS_RUNTIME is not None:
            raise RuntimeError("process installation runtime was already published")
        _PROCESS_RUNTIME = runtime


def _retain_failed_startup(
    *,
    raw_graph: Mapping[str, object],
    broker: DeviceBroker | None,
    resources: ResourceArbiter | None,
    journal: PersistentSafetyJournal | None,
) -> None:
    with _FAILED_STARTUP_LOCK:
        _FAILED_STARTUPS.append(
            _FailedVirtualStartupAuthority(
                dict(raw_graph),
                broker,
                resources,
                journal,
            )
        )


def _deployed_target() -> PulseTarget:
    """Validate and retain the pulse owner's deployed topology without projection."""

    target = load_deployed_pulse_target()
    required = {
        *_COOLING_CHANNELS,
        *_PROBE_CHANNELS,
        *_TRAP_CHANNELS,
        *_CAMERA_TRIGGER_CHANNELS,
    }
    missing = tuple(sorted(required.difference(target.by_key)))
    if missing:
        raise RuntimeError(
            f"deployed PulseTarget is missing virtual installation ports {missing}"
        )
    for key in sorted(required):
        port = target.by_key[key]
        if port.kind != PORT_DIGITAL or port.lanes != (key,):
            raise RuntimeError(
                f"virtual installation wiring {key!r} is not a one-lane digital port"
            )
    return target


def _identity_for(
    asset: InstallationAsset,
    asset_map_revision: str,
) -> PhysicalDeviceIdentity:
    if (
        asset.evidence_kind
        is not DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT
    ):
        raise ValueError("virtual assets require installation-asserted identity")
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
) -> BoundCapturePort:
    endpoint = CameraCaptureEndpoint(camera, asset.role)
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
        cleanup_operations={SafetyOperation.DISARM: endpoint.cleanup},
        verify_safe_state=endpoint.verify_safe_state,
        capability_probe=lambda: endpoint.capability_probe(current_binding()),
        close_session=lambda command: endpoint.close_session(
            current_binding(),
            command,
        ),
        interrupt_operations={SafetyOperation.DISARM: endpoint.interrupt},
    )
    return BoundCapturePort(
        broker.verify_capability(binding),
        (SafetyOperation.DISARM,),
    )


def _bind_sequencer(
    broker: DeviceBroker,
    asset: InstallationAsset,
    asset_map_revision: str,
    sequencer: VirtualSequencer,
) -> BoundPulsePort:
    endpoint = VirtualSequencerExecutionEndpoint(sequencer)
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
        cleanup_operations={SafetyOperation.SAFE_STATE: endpoint.cleanup},
        verify_safe_state=endpoint.verify_safe_state,
        capability_probe=lambda: endpoint.capability_probe(current_binding()),
        close_session=lambda command: endpoint.close_session(
            current_binding(),
            command,
        ),
        interrupt_operations={SafetyOperation.SAFE_STATE: endpoint.interrupt},
    )
    return BoundPulsePort(
        broker.verify_capability(binding),
        (),
    )


class _VirtualInstallationRuntime:
    """Virtual-only owner of the raw graph, admission, and terminal shutdown."""

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
        "_pulse_ports",
        "_raw_graph",
        "_close_order",
        "_closed_roles",
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
        pulse_ports: Mapping[str, BoundPulsePort],
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
        pulse_ports = dict(pulse_ports)
        raw_graph = dict(raw_graph)
        close_order = tuple(close_order)
        if len(close_order) != len(set(close_order)):
            raise ValueError("installation close order contains duplicate roles")
        if set(close_order) != set(raw_graph):
            raise ValueError(
                "installation close order must cover the raw graph exactly"
            )
        public_roles = set(camera_ports) | set(pulse_ports)
        if not public_roles.issubset(raw_graph):
            raise ValueError("installation ports reference roles outside the raw graph")
        if not public_roles.issubset(set(catalog)):
            raise ValueError("installation ports reference roles outside the catalog")
        if any(catalog.require(role).domain != "camera" for role in camera_ports):
            raise ValueError("camera port roles differ from catalog domains")
        if any(
            catalog.require(role).domain != "sequencer" for role in pulse_ports
        ):
            raise ValueError("pulse port roles differ from catalog domains")
        self._camera_ports = camera_ports
        self._pulse_ports = pulse_ports
        self._raw_graph = raw_graph
        self._close_order = close_order
        self._closed_roles: set[str] = set()

    @property
    def device_catalog(self) -> DeviceCatalogView:
        return self._catalog

    @property
    def runtime_instance_id(self) -> str:
        return self._runtime_instance_id

    def camera_port(self, reference: DeviceRef) -> BoundCapturePort:
        with self._lock:
            if self._state != "RUNNING":
                raise RuntimeError("installation runtime is not accepting operations")
            self._require_current_reference(reference, "camera")
            try:
                return self._camera_ports[reference.role]
            except KeyError as exc:
                raise ValueError(f"role {reference.role!r} is not a camera") from exc

    def pulse_port(self, reference: DeviceRef) -> BoundPulsePort:
        with self._lock:
            if self._state != "RUNNING":
                raise RuntimeError("installation runtime is not accepting operations")
            self._require_current_reference(reference, "sequencer")
            try:
                return self._pulse_ports[reference.role]
            except KeyError as exc:
                raise ValueError(f"role {reference.role!r} is not a sequencer") from exc

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
        # close, broker invalidation, or journal release.
        if not self._controller.shutdown(
            max(0.0, deadline - time.monotonic())
        ):
            return False
        self._broker.shutdown()
        for role in self._close_order:
            if role in self._closed_roles:
                continue
            close = getattr(self._raw_graph[role], "close", None)
            if callable(close):
                close()
            self._closed_roles.add(role)
        # The journal/arbiter authority is released only after every raw close
        # has acknowledged.  An exception above leaves this runtime retryable.
        self._resources.shutdown()
        with self._lock:
            self._state = "CLOSED"
            self._raw_graph.clear()
            self._camera_ports.clear()
            self._pulse_ports.clear()
            return True


def _catalog(
    installation_id: str,
    runtime_instance_id: str,
    assets: InstallationAssetMap,
    devices: Mapping[str, object],
) -> DeviceCatalogView:
    domains = {"camera": "camera", "sequencer": "sequencer", "trap": "trap"}
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
            if asset.role in devices and asset.role in {"camera", "sequencer"}
        ),
    )


def create_virtual_installation(
    *,
    safety_journal_path: str | Path,
    seed: int | None = 7,
) -> _VirtualInstallationRuntime:
    """Build and publish one immutable virtual graph or fail without a partial runtime."""

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("virtual installation seed must be an integer or None")
        if seed < 0:
            raise ValueError("virtual installation seed must be non-negative")
    journal_path = Path(safety_journal_path).expanduser().resolve()
    _require_no_failed_startup()
    _claim_process_composition()
    trap: VirtualAtomArray | None = None
    sequencer: VirtualSequencer | None = None
    camera: VirtualCamera | None = None
    devices: dict[str, object] = {}
    journal: PersistentSafetyJournal | None = None
    resources: ResourceArbiter | None = None
    broker: DeviceBroker | None = None
    try:
        target = _deployed_target()
        trap = VirtualAtomArray(
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
        # The trap is a private simulator model behind the camera, not an unbound
        # public physical-device role.  It remains in the exact reverse-close graph.
        assets = InstallationAssetMap.ephemeral(
            {"sequencer": sequencer, "camera": camera}
        )
        installation_id = f"installation-{assets.revision[:20]}"
        runtime_instance_id = uuid.uuid4().hex
        durable_mkdir(journal_path.parent)
        for role in ("sequencer", "camera"):
            devices[role].ensure_open()
        sequencer.set_safe_state()
        broker = DeviceBroker()
        camera_port = _bind_camera(
            broker,
            assets.require("camera", camera),
            assets.revision,
            camera,
        )
        pulse_port = _bind_sequencer(
            broker,
            assets.require("sequencer", sequencer),
            assets.revision,
            sequencer,
        )
        catalog = _catalog(
            installation_id,
            runtime_instance_id,
            assets,
            devices,
        )
        # Acquire durable authority last, after all fallible adapter construction,
        # open, identity, binding, and capability probes have succeeded.
        journal = PersistentSafetyJournal(journal_path)
        resources = ResourceArbiter(journal)
        controller = RunController(resources)
        runtime = _VirtualInstallationRuntime(
            installation_id=installation_id,
            runtime_instance_id=runtime_instance_id,
            catalog=catalog,
            resources=resources,
            broker=broker,
            controller=controller,
            camera_ports={"camera": camera_port},
            pulse_ports={"sequencer": pulse_port},
            raw_graph=devices,
            close_order=("camera", "sequencer", "trap"),
        )
        _retain_process_runtime(runtime)
        return runtime
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        authority_actions = (
            None if broker is None else broker.shutdown,
            None if camera is None else camera.close,
            None if sequencer is None else sequencer.close,
            None if trap is None else trap.close,
        )
        for action in authority_actions:
            if action is None:
                continue
            try:
                action()
            except BaseException as error:
                cleanup_errors.append(error)
        if not cleanup_errors:
            final_action = (
                resources.shutdown
                if resources is not None
                else (None if journal is None else journal.close)
            )
            if final_action is not None:
                try:
                    final_action()
                except BaseException as error:
                    cleanup_errors.append(error)
        if cleanup_errors:
            _retain_failed_startup(
                raw_graph=devices,
                broker=broker,
                resources=resources,
                journal=journal,
            )
        for error in cleanup_errors:
            try:
                primary.add_note(
                    "virtual installation startup cleanup also failed: "
                    f"{type(error).__name__}: {error}"
                )
            except BaseException:
                pass
        raise


__all__: list[str] = []
