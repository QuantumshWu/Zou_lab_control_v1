"""Connect a validated device graph and own its one runtime generation."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from zlc_neutral_atom.device_types import (
    CAPABILITY_CAMERA_SIGNAL_ASSOCIATION,
    CAPABILITY_READOUT_APPARATUS,
    dependency_capabilities,
    device_type,
    validate_installation_graph,
)
from zlc_neutral_atom.installation import (
    DeviceCatalogView,
    DeviceInfo,
    DeviceRef,
    ReadoutApparatusFacts,
)
from zlc_neutral_atom.installation_config import InstallationConfigDocument
from zlc_neutral_atom.runtime.ports import DeviceBroker
from zlc_neutral_atom.runtime.resources import ResourceArbiter
from zlc_neutral_atom.runtime.run import RunController, RunHandle, RunPlan
from zlc_storage import canonical_text


class _InstallationRuntime:
    """Connection-lifetime owner of capabilities, Runs, and reverse cleanup."""

    __slots__ = (
        "_lock",
        "_shutdown_lock",
        "_state",
        "_runtime_instance_id",
        "_catalog",
        "_resources",
        "_broker",
        "_controller",
        "_capabilities",
        "_closers",
        "_broker_closed",
        "_resources_closed",
        "_closed_instances",
        "_shutdown_diagnostics",
    )

    def __init__(
        self,
        *,
        runtime_instance_id: str,
        catalog: DeviceCatalogView,
        resources: ResourceArbiter,
        broker: DeviceBroker,
        controller: RunController,
        capabilities: Mapping[str, Mapping[str, object]],
        closers: tuple[tuple[str, Callable[[], None]], ...],
    ) -> None:
        self._lock = threading.RLock()
        self._shutdown_lock = threading.Lock()
        self._state = "RUNNING"
        self._runtime_instance_id = canonical_text(
            runtime_instance_id,
            "runtime_instance_id",
        )
        if not isinstance(catalog, DeviceCatalogView):
            raise TypeError("catalog must be DeviceCatalogView")
        if catalog.runtime_instance_id != self._runtime_instance_id:
            raise ValueError("catalog belongs to another runtime generation")
        self._catalog = catalog
        self._resources = resources
        self._broker = broker
        self._controller = controller
        copied: dict[str, Mapping[str, object]] = {}
        for instance_id, values in capabilities.items():
            info = catalog.require(instance_id)
            if set(values) != set(info.capabilities):
                raise ValueError(
                    f"capabilities for {instance_id!r} differ from its descriptor"
                )
            copied[instance_id] = MappingProxyType(dict(values))
        if set(copied) != set(catalog):
            raise ValueError("runtime capabilities must cover the catalog exactly")
        self._capabilities = copied
        closers = tuple(closers)
        closer_ids = tuple(instance_id for instance_id, _close in closers)
        if len(closer_ids) != len(set(closer_ids)) or set(closer_ids) != set(catalog):
            raise ValueError("runtime closers must cover each device instance once")
        if any(not callable(close) for _instance_id, close in closers):
            raise TypeError("runtime closers must be callable")
        self._closers = closers
        self._broker_closed = False
        self._resources_closed = False
        self._closed_instances: set[str] = set()
        self._shutdown_diagnostics: tuple[str, ...] = ()

    @property
    def device_catalog(self) -> DeviceCatalogView:
        return self._catalog

    @property
    def runtime_instance_id(self) -> str:
        return self._runtime_instance_id

    @property
    def shutdown_diagnostics(self) -> tuple[str, ...]:
        with self._lock:
            return self._shutdown_diagnostics

    def require_capability(self, reference: DeviceRef, token: str) -> object:
        """Resolve one exact capability without exposing a runtime service locator."""

        token = canonical_text(token, "capability token")
        with self._lock:
            if self._state != "RUNNING":
                raise RuntimeError("installation runtime is not accepting operations")
            info = self._require_current_reference(reference)
            if token not in info.capabilities:
                raise ValueError(
                    f"device {info.instance_id!r} does not provide {token!r}"
                )
            return self._capabilities[info.instance_id][token]

    def _require_current_reference(self, reference: DeviceRef) -> DeviceInfo:
        if not isinstance(reference, DeviceRef):
            raise TypeError("device reference must be DeviceRef")
        if reference.runtime_instance_id != self._runtime_instance_id:
            raise RuntimeError("device reference belongs to another runtime generation")
        info = self._catalog.require(reference.instance_id)
        if info.type_id != reference.type_id:
            raise RuntimeError("device reference type differs from the active instance")
        return info

    def start(self, plan: RunPlan) -> RunHandle:
        with self._lock:
            if self._state != "RUNNING":
                raise RuntimeError("installation runtime is not accepting Runs")
            return self._controller.start(plan)

    def wait(self, handle: RunHandle):
        with self._lock:
            controller = self._controller
        return controller.wait(handle)

    def wait_until_released(
        self,
        run_ids: tuple[str, ...],
        *,
        timeout: float | None = None,
    ) -> bool:
        with self._lock:
            resources = self._resources
        return resources.wait_until_released(run_ids, timeout=timeout)

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
        if not self._controller.shutdown(max(0.0, deadline - time.monotonic())):
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
                failures.append(f"device broker: {type(error).__name__}: {error}")
            else:
                self._broker_closed = True
        for instance_id, close in reversed(self._closers):
            if instance_id in self._closed_instances:
                continue
            try:
                close()
            except BaseException as error:
                failures.append(
                    f"{instance_id}: {type(error).__name__}: {error}"
                )
            else:
                self._closed_instances.add(instance_id)
        if not failures and not self._resources_closed:
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
            self._capabilities.clear()
        return True


@dataclass(frozen=True, slots=True)
class _InstallationComposition:
    """Runtime plus the two capability-free projections used by logic composition."""

    runtime: _InstallationRuntime
    readout_apparatus_facts: tuple[ReadoutApparatusFacts, ...] = ()
    camera_signal_association_authorities: tuple[tuple[str, object], ...] = ()


def create_installation(
    document: InstallationConfigDocument,
    *,
    required_pulse_document: object | None = None,
) -> _InstallationComposition:
    """Preflight completely, then connect topologically and publish atomically."""

    ordered = validate_installation_graph(document)
    runtime_instance_id = uuid.uuid4().hex
    resources = ResourceArbiter()
    broker = DeviceBroker()
    connected: dict[str, Mapping[str, object]] = {}
    closers: list[tuple[str, Callable[[], None]]] = []
    try:
        for instance in ordered:
            descriptor = device_type(instance.type_id)
            dependencies = dependency_capabilities(instance, connected)
            capabilities, close = descriptor.factory(
                instance,
                dependencies,
                broker,
                required_pulse_document,
            )
            capabilities = dict(capabilities)
            if set(capabilities) != set(descriptor.capabilities):
                raise RuntimeError(
                    f"factory {descriptor.type_id!r} published capabilities "
                    "different from its descriptor"
                )
            if any(value is None for value in capabilities.values()):
                raise RuntimeError("device factory published an empty capability")
            if not callable(close):
                raise TypeError("device factory close result must be callable")
            connected[instance.instance_id] = MappingProxyType(capabilities)
            closers.append((instance.instance_id, close))

        catalog = DeviceCatalogView(
            runtime_instance_id,
            tuple(
                DeviceInfo(
                    ref=DeviceRef(
                        runtime_instance_id,
                        instance.instance_id,
                        instance.type_id,
                        instance.role,
                    ),
                    domain=device_type(instance.type_id).domain,
                    capabilities=device_type(instance.type_id).capabilities,
                    resource_key=f"device/{instance.instance_id}",
                )
                for instance in document.devices
            ),
        )
        runtime = _InstallationRuntime(
            runtime_instance_id=runtime_instance_id,
            catalog=catalog,
            resources=resources,
            broker=broker,
            controller=RunController(resources),
            capabilities=connected,
            closers=tuple(closers),
        )
        apparatus = tuple(
            value
            for instance in document.devices
            for value in (
                connected[instance.instance_id].get(CAPABILITY_READOUT_APPARATUS),
            )
            if value is not None
        )
        if any(not isinstance(value, ReadoutApparatusFacts) for value in apparatus):
            raise TypeError("readout apparatus capability has the wrong value type")
        authorities = tuple(
            (instance.role, value)
            for instance in document.devices
            for value in (
                connected[instance.instance_id].get(
                    CAPABILITY_CAMERA_SIGNAL_ASSOCIATION
                ),
            )
            if value is not None
        )
        return _InstallationComposition(
            runtime=runtime,
            readout_apparatus_facts=apparatus,
            camera_signal_association_authorities=authorities,
        )
    except BaseException as primary:
        failures: list[BaseException] = []
        try:
            broker.shutdown()
        except BaseException as error:
            failures.append(error)
        for _instance_id, close in reversed(closers):
            try:
                close()
            except BaseException as error:
                failures.append(error)
        try:
            resources.shutdown()
        except BaseException as error:
            failures.append(error)
        for error in failures:
            primary.add_note(
                "installation startup cleanup also failed: "
                f"{type(error).__name__}: {error}"
            )
        raise


__all__ = ["_InstallationComposition", "_InstallationRuntime", "create_installation"]
