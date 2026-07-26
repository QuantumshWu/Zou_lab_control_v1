"""Low-level runtime, graph, catalog, and composition values for installations."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping

from zlc_neutral_atom.installation import (
    DeviceCatalogView,
    DeviceInfo,
    DeviceRef,
    ReadoutApparatusFacts,
)
from zlc_neutral_atom.installation_plan import InstallationDevicePlan
from zlc_neutral_atom.devices.rf import BoundRfTablePort
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.camera.monitor import BoundCameraMonitorPort
from zlc_neutral_atom.runtime.ports import DeviceBroker
from zlc_neutral_atom.runtime.resources import DeviceIdentityEvidenceKind, PhysicalDeviceIdentity, ResourceArbiter
from zlc_neutral_atom.runtime.run import RunController, RunHandle, RunPlan
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_storage import canonical_digest

from .installation_assets import InstallationAsset, InstallationAssetMap

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


class _InstallationRuntime:
    """Connection-lifetime owner of one immutable installation graph."""

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
        "_rf_ports",
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
        rf_ports: Mapping[str, BoundRfTablePort],
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
        rf_ports = dict(rf_ports)
        raw_graph = dict(raw_graph)
        close_order = tuple(close_order)
        if len(close_order) != len(set(close_order)):
            raise ValueError("installation close order contains duplicate roles")
        if set(close_order) != set(raw_graph):
            raise ValueError(
                "installation close order must cover the raw graph exactly"
            )
        public_roles = (
            set(camera_ports)
            | set(camera_monitor_ports)
            | set(pulse_ports)
            | set(rf_ports)
        )
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
        if any(catalog.require(role).domain != "rf" for role in rf_ports):
            raise ValueError("RF port roles differ from catalog domains")
        self._camera_ports = camera_ports
        self._camera_monitor_ports = camera_monitor_ports
        self._pulse_ports = pulse_ports
        self._rf_ports = rf_ports
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

    def rf_port(self, reference: DeviceRef) -> BoundRfTablePort:
        with self._lock:
            if self._state != "RUNNING":
                raise RuntimeError("installation runtime is not accepting operations")
            self._require_current_reference(reference, "rf")
            try:
                return self._rf_ports[reference.role]
            except KeyError as exc:
                raise ValueError(
                    f"RF role {reference.role!r} has no synchronized table Port"
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
            self._rf_ports.clear()
            return True


@dataclass(frozen=True, slots=True)
class _InstallationComposition:
    """One runtime plus explicit capability-free apparatus observations."""

    runtime: _InstallationRuntime
    readout_apparatus_facts: tuple[ReadoutApparatusFacts, ...] = ()
    camera_signal_association_authorities: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, _InstallationRuntime):
            raise TypeError("runtime must be _InstallationRuntime")
        facts = tuple(self.readout_apparatus_facts)
        if any(not isinstance(item, ReadoutApparatusFacts) for item in facts):
            raise TypeError(
                "readout_apparatus_facts must contain ReadoutApparatusFacts"
            )
        roles = tuple(item.camera_role for item in facts)
        if len(roles) != len(set(roles)):
            raise ValueError("readout apparatus camera roles must be unique")
        catalog = self.runtime.device_catalog
        for item in facts:
            if catalog.require(item.camera_role).domain != "camera":
                raise ValueError("readout apparatus camera role is not a camera")
            if catalog.require(item.sequencer_role).domain != "sequencer":
                raise ValueError(
                    "readout apparatus sequencer role is not a sequencer"
                )
        object.__setattr__(self, "readout_apparatus_facts", facts)
        authorities = tuple(self.camera_signal_association_authorities)
        roles: list[str] = []
        required_methods = (
            "arm_signal_event_association",
            "bind_signal_event_association",
            "finish_signal_event_association",
            "cancel_signal_event_association",
        )
        for item in authorities:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "camera signal association entries must be (role, authority) tuples"
                )
            role, authority = item
            if not isinstance(role, str) or not role or role.strip() != role:
                raise ValueError("camera signal association role must be canonical text")
            if catalog.require(role).domain != "camera":
                raise ValueError(
                    "camera signal association role is not a camera"
                )
            if any(not callable(getattr(authority, name, None)) for name in required_methods):
                raise TypeError(
                    "camera signal association authority has an incomplete contract"
                )
            roles.append(role)
        if len(roles) != len(set(roles)):
            raise ValueError("camera signal association roles must be unique")
        object.__setattr__(
            self,
            "camera_signal_association_authorities",
            authorities,
        )


def _catalog(
    installation_id: str,
    runtime_instance_id: str,
    assets: InstallationAssetMap,
    devices: Mapping[str, object],
    plan: tuple[InstallationDevicePlan, ...],
) -> DeviceCatalogView:
    planned = {item.role: item for item in plan}
    by_role = {asset.role: asset for asset in assets.assets}
    if set(by_role) != set(planned):
        raise RuntimeError(
            "composed installation assets differ from its public device plan"
        )
    for role, item in planned.items():
        if role not in devices:
            raise RuntimeError(f"installation did not compose planned role {role!r}")
        actual_kind = by_role[role].adapter_kind
        if actual_kind != item.adapter_kind:
            raise RuntimeError(
                f"installation role {role!r} built adapter {actual_kind!r}, "
                f"expected {item.adapter_kind!r}"
            )
    return DeviceCatalogView(
        installation_id,
        runtime_instance_id,
        0,
        tuple(
            DeviceInfo(
                DeviceRef(installation_id, runtime_instance_id, item.role),
                item.domain,
                by_role[item.role].adapter_kind,
                str(by_role[item.role].resource_key),
            )
            for item in plan
        ),
    )
