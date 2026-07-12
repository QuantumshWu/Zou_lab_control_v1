"""Composition-owned adapters for the temporary neutral-atom LogicNode island.

The generic fence lives in :mod:`zlc_workbench.legacy_runtime`.  This module is
the only bridge that knows the concrete legacy device classes.  Every supported
device has an explicit identity and terminal-safety recipe; an unknown concrete
adapter is rejected during composition instead of being guessed with ``getattr``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import queue
import threading
import time
import uuid
import weakref
from typing import Callable, Mapping

from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera
from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera
from Zou_lab_control.neutral_atom.devices.base import BaseDevice
from Zou_lab_control.neutral_atom.devices.sequencer import (
    ManualSequencer,
    RemoteSequencer,
)
from Zou_lab_control.neutral_atom.devices.virtual import (
    VirtualCamera,
    VirtualLaser,
    VirtualMotCamera,
    VirtualRF,
    VirtualSequencer,
    VirtualTrapArray,
)
from zlc_neutral_atom.runtime import (
    BoundMeasurement,
    CleanupStepAck,
    ConnectionEstablishmentLease,
    DeviceBroker,
    DeviceIdentityAck,
    DeviceIdentityEvidenceKind,
    MemoryQuarantineJournal,
    PersistentSafetyJournal,
    ResourceArbiter,
    ResourceBusy,
    ResourceKey,
    ResourceQuarantined,
    RunController,
    SafeStateAck,
    SafetyOperation,
)
from zlc_storage import canonical_digest

from .asset_map import InstallationAsset, InstallationAssetMap
from .camera_capture import (
    CameraCaptureBindingRequest,
    CameraCaptureEndpoint,
    bind_camera_measurement as _bind_camera_measurement,
)
from .legacy_runtime import (
    LegacyDeviceRegistration,
    LegacyDeviceRegistry,
    LegacyRuntimeFence,
)


_CONNECTION_RECEIPTS: "weakref.WeakKeyDictionary[object, tuple[object, str]]" = (
    weakref.WeakKeyDictionary()
)
_CONNECTION_RECEIPTS_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class PreparedLegacyDeviceSet:
    """Validated, unpublished registry for one exact candidate DeviceSet."""

    device_set: object
    registrations: tuple[LegacyDeviceRegistration, ...]
    registry: LegacyDeviceRegistry


class _SequencerCommand:
    def __init__(self, name: str, argument: object = None) -> None:
        self.name = str(name)
        self.argument = argument
        self.done = threading.Event()
        self.result: object = None
        self.error: BaseException | None = None


class _LegacySequencerSessionNode:
    """One prepared/firing sequencer session held under a single fenced Run."""

    def __init__(self, sequencer, payload) -> None:
        self.sequencer = sequencer
        self.payload = payload
        self._authority: object | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._commands: "queue.Queue[_SequencerCommand]" = queue.Queue()
        self._prepared = threading.Event()
        self._prepared_result: object = None
        self._prepare_error: BaseException | None = None
        self._terminal_error: BaseException | None = None

    def occupied_devices(self):
        return (self.sequencer,)

    def referenced_devices(self):
        return (self.sequencer,)

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def _execution_error(self) -> BaseException | None:
        return self._prepare_error or self._terminal_error

    def _bind_runtime_start_authority(self, authority: object) -> None:
        if authority is None:
            raise ValueError("runtime start authority cannot be None")
        if self._authority is not None and self._authority is not authority:
            raise RuntimeError("sequencer session is already bound to another authority")
        self._authority = authority

    def _start_from_runtime(self, authority: object):
        if self._authority is not authority:
            raise RuntimeError("sequencer session start capability is not current")
        return self._start_impl()

    def start(self):
        raise RuntimeError("sequencer sessions require LegacyRuntimeFence authority")

    def _start_impl(self):
        self._stop.clear()
        self._prepared.clear()

        def owner() -> None:
            try:
                self._prepared_result = self.sequencer.prepare(self.payload)
            except BaseException as exc:
                self._prepare_error = exc
            finally:
                self._prepared.set()
            try:
                if self._prepare_error is None:
                    self._serve_commands()
            finally:
                try:
                    self.sequencer.set_safe_state()
                except BaseException as exc:
                    self._terminal_error = exc

        self._thread = threading.Thread(
            target=owner,
            name=f"zlc-legacy-sequencer-{id(self.sequencer):x}",
            daemon=False,
        )
        self._thread.start()
        return self

    def _serve_commands(self) -> None:
        while not self._stop.is_set():
            try:
                command = self._commands.get(timeout=0.02)
            except queue.Empty:
                continue
            try:
                if command.name == "fire":
                    command.result = self.sequencer.fire(command.argument)
                elif command.name == "wait_done":
                    command.result = self.sequencer.wait_done(
                        timeout=command.argument, stop=self._stop
                    )
                elif command.name == "scan_progress":
                    command.result = self.sequencer.scan_progress()
                elif command.name == "snapshot":
                    command.result = self.sequencer.snapshot()
                else:
                    raise RuntimeError(f"unknown sequencer session command {command.name!r}")
            except BaseException as exc:
                command.error = exc
            finally:
                command.done.set()

    def wait_prepared(self) -> object:
        self._prepared.wait()
        if self._prepare_error is not None:
            raise self._prepare_error
        return self._prepared_result

    def command(self, name: str, argument: object = None) -> object:
        if not self.running or self._prepare_error is not None:
            raise RuntimeError("sequencer session is not prepared and running")
        command = _SequencerCommand(name, argument)
        self._commands.put(command)
        command.done.wait()
        if command.error is not None:
            raise command.error
        return command.result

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
        if thread is not None and thread.is_alive():
            return False
        self._thread = None
        if self._terminal_error is not None:
            raise self._terminal_error
        return True


class LegacySequencerFacade:
    """Session-bound sequencer surface with no raw drive capability escape."""

    managed_installation_authority = True

    def __init__(self, runtime: "LegacyNeutralAtomRuntime", sequencer) -> None:
        self._runtime = runtime
        self._sequencer = sequencer
        self._lock = threading.RLock()
        self._active_node: _LegacySequencerSessionNode | None = None
        self._active_handle = None

    @property
    def _authority_device(self):
        return self._sequencer

    def __getattr__(self, name: str):
        if name in {
            "channels",
            "clock_hz",
            "port_catalog",
            "host",
            "port",
            "SCAN_PROGRESS_IDLE",
        }:
            return getattr(self._sequencer, name)
        raise AttributeError(name)

    def open(self):
        self._runtime.ensure_connections((self._sequencer,))
        return self

    def close(self) -> None:
        self.set_safe_state()

    def prepare(self, payload):
        if self._runtime.owns_current_thread(self._sequencer):
            return self._sequencer.prepare(payload)
        self._stop_active("sequencer re-prepare")
        node = _LegacySequencerSessionNode(self._sequencer, payload)
        handle = self._runtime.start(node)
        try:
            handle.wait_started()
            result = node.wait_prepared()
        except BaseException:
            self._runtime.fence.stop(
                node, timeout=2.0, reason="sequencer prepare failed"
            )
            raise
        with self._lock:
            self._active_node = node
            self._active_handle = handle
        return result

    def fire(self, sequence_payload=None):
        if self._runtime.owns_current_thread(self._sequencer):
            return self._sequencer.fire(sequence_payload)
        node = self._require_active()
        try:
            return node.command("fire", sequence_payload)
        except BaseException:
            self._stop_active("sequencer fire failed")
            raise

    def wait_done(self, timeout: float | None = None, *, stop=None) -> bool:
        if self._runtime.owns_current_thread(self._sequencer):
            return bool(self._sequencer.wait_done(timeout=timeout, stop=stop))
        if stop is not None and stop.is_set():
            self._stop_active("sequencer wait cancelled")
            return False
        node = self._require_active()
        try:
            done = bool(node.command("wait_done", timeout))
        except BaseException:
            self._stop_active("sequencer wait failed")
            raise
        if done:
            self._stop_active("finite sequencer program completed")
        return done

    def set_safe_state(self) -> None:
        if self._runtime.owns_current_thread(self._sequencer):
            self._sequencer.set_safe_state()
            return
        if self._stop_active("sequencer safe state requested"):
            return
        if not bool(getattr(self._sequencer, "is_open", True)):
            # A closed connection cannot own an active program.  Opening hardware merely
            # to issue SAFE would create a new generation outside recovery semantics.
            return
        self._runtime._run_hardware_call(
            (self._sequencer,),
            self._sequencer.set_safe_state,
            name="sequencer-safe-state",
        )

    def abort(self) -> None:
        self.set_safe_state()

    def scan_progress(self) -> dict:
        if self._runtime.owns_current_thread(self._sequencer):
            return dict(self._sequencer.scan_progress())
        node = self._active()
        if node is not None:
            return dict(node.command("scan_progress"))
        return dict(
            self._runtime._run_hardware_call(
                (self._sequencer,),
                self._sequencer.scan_progress,
                name="sequencer-scan-progress",
                observe=True,
            )
        )

    def snapshot(self) -> dict:
        if self._runtime.owns_current_thread(self._sequencer):
            return dict(self._sequencer.snapshot())
        node = self._active()
        if node is not None:
            return dict(node.command("snapshot"))
        return dict(
            self._runtime._run_hardware_call(
                (self._sequencer,),
                self._sequencer.snapshot,
                name="sequencer-snapshot",
                observe=True,
            )
        )

    def _active(self) -> _LegacySequencerSessionNode | None:
        with self._lock:
            node = self._active_node
            handle = self._active_handle
        if node is None or handle is None:
            return None
        if handle.snapshot().state.terminal:
            with self._lock:
                if self._active_node is node:
                    self._active_node = None
                    self._active_handle = None
            return None
        return node

    def _require_active(self) -> _LegacySequencerSessionNode:
        node = self._active()
        if node is None:
            raise RuntimeError("sequencer fire/wait requires a prepared session")
        return node

    def _stop_active(self, reason: str) -> bool:
        node = self._active()
        if node is None:
            return False
        receipt = self._runtime.fence.stop(node, timeout=2.0, reason=reason)
        if not receipt.terminated:
            raise RuntimeError("sequencer owner is still cancelling")
        with self._lock:
            if self._active_node is node:
                self._active_node = None
                self._active_handle = None
        return True


def default_safety_journal_path() -> Path:
    """Return the machine-installation ledger, independent of artifact storage."""

    root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "state")
    return root / "ZouLabControl" / "safety" / "neutral-atom-runtime.zlcj"


def default_asset_map_path() -> Path:
    """Return the machine installation AssetMap, never an experiment repository path."""

    configured = os.environ.get("ZLC_ASSET_MAP")
    if configured:
        return Path(configured)
    root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "state")
    return root / "ZouLabControl" / "installation" / "asset-map.json"


class LegacyNeutralAtomRuntime:
    """One installation authority shared by legacy and target runtime callers."""

    def __init__(
        self,
        device_set,
        *,
        safety_journal_path: str | Path | None = None,
        asset_map: InstallationAssetMap | Mapping[str, object] | str | Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._lifecycle = threading.Condition(self._lock)
        self._closed = False
        self._closing = False
        self._establishments = 0
        self._safety_journal_path = (
            None if safety_journal_path is None else Path(safety_journal_path)
        )
        self.asset_map = _resolve_asset_map(device_set, asset_map)
        registrations = registrations_for(device_set, asset_map=self.asset_map)
        self._persistent = _requires_persistent_safety(registrations)
        journal = (
            PersistentSafetyJournal(
                self._safety_journal_path or default_safety_journal_path()
            )
            if self._persistent
            else MemoryQuarantineJournal()
        )
        self.resources = ResourceArbiter(journal)
        self.broker = DeviceBroker()
        self.controller = RunController(self.resources)
        self.registry = _registry_from(registrations, self.broker)
        self.fence = LegacyRuntimeFence(self.controller, self.registry)
        self._device_set = device_set
        self._pulse_facades: dict[int, LegacySequencerFacade] = {}
        self._authority_local = threading.local()

    @property
    def persistent(self) -> bool:
        return self._persistent

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def persistence_required(
        device_set,
        *,
        asset_map: InstallationAssetMap | Mapping[str, object] | str | Path | None = None,
    ) -> bool:
        return _requires_persistent_safety(
            registrations_for(device_set, asset_map=asset_map)
        )

    def prepare_device_set(self, device_set) -> PreparedLegacyDeviceSet:
        """Validate and stage one registry without publishing a new generation."""

        registrations = registrations_for(device_set, asset_map=self.asset_map)
        return PreparedLegacyDeviceSet(
            device_set,
            registrations,
            _registry_from(registrations, self.broker),
        )

    def begin_device_transition(self) -> object:
        with self._lock:
            self._ensure_open()
            if self._establishments:
                raise RuntimeError(
                    "cannot replace a device generation during connection establishment"
                )
            return self.fence.begin_device_transition()

    def start(self, node):
        """Establish every referenced connection before delegating to the generic fence."""

        referenced = tuple(getattr(node, "referenced_devices", lambda: ())())
        self.ensure_connections(referenced)
        return self.fence.start(node)

    def stop(self, node, **kwargs):
        return self.fence.stop(node, **kwargs)

    def handle_for(self, node):
        return self.fence.handle_for(node)

    def _run_hardware_call(
        self,
        devices,
        callback: Callable[[], object],
        *,
        name: str,
        observe: bool = False,
    ) -> object:
        """Route one old notebook/GUI drive verb through the installation Run authority."""

        if not isinstance(observe, bool):
            raise TypeError("observe must be bool")
        concrete = tuple(device for device in devices if device is not None)
        requested = frozenset(id(device) for device in concrete)
        active = getattr(self._authority_local, "device_ids", None)
        if active is not None:
            if not requested.issubset(active):
                raise RuntimeError("nested hardware call requested an undeclared device")
            return callback()
        with self._lock:
            self._ensure_open()
            fence = self.fence

        self.ensure_connections(concrete)

        def owned_call():
            self._authority_local.device_ids = requested
            try:
                return callback()
            finally:
                del self._authority_local.device_ids

        return fence.call(concrete, owned_call, name=name, observe=observe)

    def ensure_connections(self, devices) -> bool:
        """Establish closed legacy connections before identity-bound Run admission.

        The transition gate prevents a config swap or another connection transaction from
        racing the open.  It is deliberately released before the Run claim: if a swap wins
        that narrow race, ``fence.start`` rejects while admission is closed and the hardware
        callback never executes.  Connections remain application-owned and may be reused.
        """

        return self._ensure_connections(
            devices,
            registry=self.registry,
            device_set=self._device_set,
        )

    def bind_camera_measurement(
        self,
        request: CameraCaptureBindingRequest,
    ) -> BoundMeasurement:
        """Resolve a typed camera request without exposing the installation raw graph."""

        if not isinstance(request, CameraCaptureBindingRequest):
            raise TypeError("request must be CameraCaptureBindingRequest")
        with self._lock:
            self._ensure_open()
            device_set = self._device_set
            registry = self.registry
            devices = dict(getattr(device_set, "devices", {}) or {})
            try:
                camera = devices[request.role]
            except KeyError as exc:
                raise KeyError(
                    f"camera role {request.role!r} is absent from installation"
                ) from exc
        self._ensure_connections(
            (camera,),
            registry=registry,
            device_set=device_set,
        )
        return _bind_camera_measurement(device_set, registry, request)

    def _ensure_connections(self, devices, *, registry, device_set) -> bool:
        if not isinstance(registry, LegacyDeviceRegistry):
            raise TypeError("registry must be LegacyDeviceRegistry")

        unique: list[object] = []
        seen: set[int] = set()
        for device in devices:
            if device is None or id(device) in seen:
                continue
            seen.add(id(device))
            unique.append(device)
        concrete = tuple(unique)
        if not concrete:
            return False
        with self._lock:
            self._ensure_open()
            registrations = tuple(registry.registration_for(device) for device in concrete)
            for registration in registrations:
                if registration.admission_check is not None:
                    registration.admission_check()
            needed = tuple(
                registration
                for registration in registrations
                if (
                    not bool(getattr(registration.device, "is_open", True))
                    or not registry.has_binding(registration.device)
                )
            )
            if not needed:
                return False
            if self._closing:
                raise RuntimeError("runtime shutdown has closed connection establishment")
            self._establishments += 1

        ordered = self._dependency_order(
            tuple(item.device for item in needed), device_set=device_set
        )
        by_device = {id(item.device): item for item in needed}
        keys = tuple(sorted(item.key for item in needed))
        lease = None
        try:
            lease = self.resources.begin_connection_establishment(keys)
            if isinstance(lease, ResourceBusy):
                raise RuntimeError(
                    f"connection establishment blocked by {lease.conflicting_run_id}"
                )
            if isinstance(lease, ResourceQuarantined):
                raise RuntimeError(
                    f"connection establishment refused for quarantined {lease.requested.key}: "
                    f"{lease.reason}; {lease.recovery_action}"
                )
            if not isinstance(lease, ConnectionEstablishmentLease):
                raise RuntimeError("invalid connection establishment result")

            changed = False
            for device in ordered:
                registration = by_device[id(device)]
                was_open = bool(getattr(device, "is_open", True))
                try:
                    if not was_open:
                        ensure_open = getattr(device, "ensure_open", None)
                        if not callable(ensure_open):
                            raise TypeError(
                                f"{type(device).__name__} must provide ensure_open()"
                            )
                        ensure_open()
                    if not bool(getattr(device, "is_open", True)):
                        raise RuntimeError(
                            f"{type(device).__name__}.ensure_open() returned without a live connection"
                        )
                    registry.establish(device)
                    changed = True
                except BaseException:
                    # A connection that cannot establish trusted identity is not retained,
                    # even when an external caller opened it before this transaction.
                    try:
                        device.close()
                    except BaseException:
                        pass
                    raise
            return changed
        finally:
            if isinstance(lease, ConnectionEstablishmentLease):
                lease.close()
            with self._lifecycle:
                self._establishments -= 1
                self._lifecycle.notify_all()

    def ensure_device_set_connections(self, device_set) -> bool:
        names = tuple(device_set._open_order())
        return self.ensure_connections(tuple(device_set.require(name) for name in names))

    def ensure_prepared_device_set_connections(
        self, prepared: PreparedLegacyDeviceSet
    ) -> bool:
        """Establish a staged set while its registry is still unpublished."""

        if not isinstance(prepared, PreparedLegacyDeviceSet):
            raise TypeError("prepared must be PreparedLegacyDeviceSet")
        names = tuple(prepared.device_set._open_order())
        devices = tuple(prepared.device_set.require(name) for name in names)
        return self._ensure_connections(
            devices,
            registry=prepared.registry,
            device_set=prepared.device_set,
        )

    def _dependency_order(
        self, devices: tuple[object, ...], *, device_set=None
    ) -> tuple[object, ...]:
        owner_set = self._device_set if device_set is None else device_set
        requested = {id(device): device for device in devices}
        ordered: list[object] = []
        for name in tuple(owner_set._open_order()):
            device = owner_set.require(name)
            selected = requested.pop(id(device), None)
            if selected is not None:
                ordered.append(selected)
        ordered.extend(requested.values())
        return tuple(ordered)

    def owns_current_thread(self, device: object) -> bool:
        active = getattr(self._authority_local, "device_ids", None)
        return active is not None and id(device) in active

    def pulse_facade(self, sequencer) -> LegacySequencerFacade:
        """Return the sole command facade for one registered sequencer connection."""

        with self._lock:
            self._ensure_open()
            self.registry.registration_for(sequencer)
            lookup = id(sequencer)
            existing = self._pulse_facades.get(lookup)
            if existing is not None and existing._sequencer is sequencer:
                return existing
            facade = LegacySequencerFacade(self, sequencer)
            self._pulse_facades[lookup] = facade
            return facade

    def commit_device_transition(
        self,
        token: object,
        device_set,
        prepared: PreparedLegacyDeviceSet,
        *,
        publish: Callable[[], None],
    ) -> None:
        """Install registry and session-visible state under one closed admission gate."""

        if not callable(publish):
            raise TypeError("publish must be callable")
        if not isinstance(prepared, PreparedLegacyDeviceSet):
            raise TypeError("prepared must be PreparedLegacyDeviceSet")
        if prepared.device_set is not device_set:
            raise ValueError("prepared registry belongs to a different DeviceSet")
        with self._lock:
            self._ensure_open()
            registry = prepared.registry
            previous_registry = self.registry
            previous_device_set = self._device_set
            previous_pulse_facades = self._pulse_facades

            def publish_all() -> None:
                try:
                    self.registry = registry
                    self._device_set = device_set
                    self._pulse_facades = {}
                    publish()
                except BaseException:
                    self.registry = previous_registry
                    self._device_set = previous_device_set
                    self._pulse_facades = previous_pulse_facades
                    raise

            self.fence.commit_device_transition(token, registry, publish=publish_all)

    def abort_device_transition(self, token: object) -> None:
        with self._lock:
            self._ensure_open()
            self.fence.abort_device_transition(token)

    def shutdown(self, *, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._lifecycle:
            if self._closed:
                return True
            self._closing = True
            while self._establishments:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._lifecycle.wait(remaining)
            receipts = self.fence.stop_all(timeout=timeout)
            if any(not receipt.terminated for receipt in receipts):
                return False
            self.resources.shutdown()
            self._closed = True
            return True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("legacy neutral-atom runtime is shut down")


def _registry_from(
    registrations: tuple[LegacyDeviceRegistration, ...],
    broker: DeviceBroker,
) -> LegacyDeviceRegistry:
    registry = LegacyDeviceRegistry(broker)
    for registration in registrations:
        registry.register(registration)
    return registry


_EXPLICIT_DEVICE_TYPES = {
    QCMOSCamera,
    PylonCamera,
    VirtualCamera,
    VirtualMotCamera,
    VirtualLaser,
    VirtualRF,
    VirtualTrapArray,
    ManualSequencer,
    RemoteSequencer,
    VirtualSequencer,
}

_EPHEMERAL_DEVICE_TYPES = {
    VirtualCamera,
    VirtualMotCamera,
    VirtualLaser,
    VirtualRF,
    VirtualTrapArray,
    ManualSequencer,
    VirtualSequencer,
}


def _resolve_asset_map(
    device_set,
    value: InstallationAssetMap | Mapping[str, object] | str | Path | None,
) -> InstallationAssetMap:
    if isinstance(value, InstallationAssetMap):
        return value
    if isinstance(value, Mapping):
        return InstallationAssetMap.from_mapping(value)
    if isinstance(value, (str, Path)):
        return InstallationAssetMap.load(value)
    if value is not None:
        raise TypeError("asset_map must be InstallationAssetMap, mapping, path, or None")
    devices = dict(getattr(device_set, "devices", {}) or {})
    if all(type(device) in _EPHEMERAL_DEVICE_TYPES for device in devices.values()):
        return InstallationAssetMap.ephemeral(devices)
    return InstallationAssetMap.load(default_asset_map_path())


def registrations_for(
    device_set,
    *,
    asset_map: InstallationAssetMap | Mapping[str, object] | str | Path | None = None,
) -> tuple[LegacyDeviceRegistration, ...]:
    """Bind every concrete legacy device to an installation-owned asset entry."""

    devices = dict(getattr(device_set, "devices", {}) or {})
    config = dict(getattr(device_set, "config", {}) or {})
    forbidden = {"resource_key", "stable_device_identity", "asset_map_revision"}
    for role, raw_entry in config.items():
        entry = dict(raw_entry or {})
        params = dict(entry.get("params") or {})
        attempted = sorted(forbidden & (set(entry) | set(params)))
        if attempted:
            raise ValueError(
                f"experiment config role {role!r} cannot define installation-owned "
                f"identity fields {attempted}"
            )
    for role, device in sorted(devices.items()):
        if not isinstance(device, BaseDevice) or type(device) not in _EXPLICIT_DEVICE_TYPES:
            raise TypeError(
                "legacy runtime has no explicit identity/safety adapter for "
                f"{type(device).__module__}.{type(device).__qualname__} at role {role!r}"
            )
    resolved_map = _resolve_asset_map(device_set, asset_map)
    registrations = []
    seen: dict[int, object] = {}
    for role in sorted(devices):
        device = devices[role]
        lookup = id(device)
        previous = seen.get(lookup)
        if previous is not None:
            raise ValueError(
                f"one device object cannot occupy both AssetMap roles {previous!r} and {role!r}"
            )
        seen[lookup] = device
        asset = resolved_map.require(role, device)
        registrations.append(
            _registration_for(role, device, asset, resolved_map.revision)
        )
    return tuple(registrations)


def _match_live_identity(asset: InstallationAsset, stable_identity: str) -> None:
    if asset.expected_identity != stable_identity:
        raise RuntimeError(
            f"AssetMap asset {asset.asset_id!r} expected identity "
            f"{asset.expected_identity!r}, live readback is {stable_identity!r}"
        )


def _require_evidence_kind(
    asset: InstallationAsset,
    expected: DeviceIdentityEvidenceKind,
) -> None:
    if asset.evidence_kind is not expected:
        raise RuntimeError(
            f"AssetMap asset {asset.asset_id!r} requires evidence kind "
            f"{asset.evidence_kind.value}, adapter proves {expected.value}"
        )


def _registration_for(
    role: str,
    device: object,
    asset: InstallationAsset,
    asset_map_revision: str,
):
    key = asset.resource_key
    if type(device) is QCMOSCamera:
        identity_probe = _qcmos_identity_probe(device, asset, asset_map_revision)
        cleanup, verify = _qcmos_safe_contract(device, key)
        endpoint = CameraCaptureEndpoint(device, role)
        return LegacyDeviceRegistration(
            device,
            key,
            identity_probe,
            {SafetyOperation.DISARM: cleanup},
            (SafetyOperation.DISARM,),
            verify,
            target_endpoint=endpoint.target_endpoint,
        )
    if type(device) is PylonCamera:
        identity_probe = _pylon_identity_probe(device, asset, asset_map_revision)
        cleanup, verify = _pylon_safe_contract(device, key)
        endpoint = CameraCaptureEndpoint(device, role)
        return LegacyDeviceRegistration(
            device,
            key,
            identity_probe,
            {SafetyOperation.DISARM: cleanup},
            (SafetyOperation.DISARM,),
            verify,
            target_endpoint=endpoint.target_endpoint,
        )
    if type(device) in {VirtualCamera, VirtualMotCamera}:
        identity_probe = _installation_identity_probe(asset, asset_map_revision)
        cleanup, verify = _virtual_camera_safe_contract(device, key)
        endpoint = CameraCaptureEndpoint(device, role)
        return LegacyDeviceRegistration(
            device,
            key,
            identity_probe,
            {SafetyOperation.DISARM: cleanup},
            (SafetyOperation.DISARM,),
            verify,
            target_endpoint=endpoint.target_endpoint,
        )
    if type(device) is VirtualSequencer:
        identity_probe = _installation_identity_probe(asset, asset_map_revision)
        cleanup, verify = _virtual_sequencer_safe_contract(device, key)
        return LegacyDeviceRegistration(
            device,
            key,
            identity_probe,
            {SafetyOperation.SAFE_STATE: cleanup},
            (SafetyOperation.SAFE_STATE,),
            verify,
        )
    if type(device) is ManualSequencer:
        return LegacyDeviceRegistration(
            device,
            key,
            _installation_identity_probe(asset, asset_map_revision),
            {},
            (),
            None,
            hazardous=False,
        )
    if type(device) is RemoteSequencer:
        cleanup = _remote_sequencer_cleanup(device, key)
        return LegacyDeviceRegistration(
            device,
            key,
            _remote_sequencer_identity_probe(device, asset, asset_map_revision),
            {SafetyOperation.SAFE_STATE: cleanup},
            (SafetyOperation.SAFE_STATE,),
            _remote_sequencer_unqualified_verifier(key),
            admission_check=_remote_sequencer_no_go(key),
        )
    if type(device) in {VirtualLaser, VirtualRF, VirtualTrapArray}:
        return LegacyDeviceRegistration(
            device,
            key,
            _installation_identity_probe(asset, asset_map_revision),
            {},
            (),
            None,
            hazardous=False,
        )
    raise TypeError(
        f"legacy runtime has no explicit identity/safety adapter for {type(device).__name__}"
    )


def _installation_identity_probe(
    asset: InstallationAsset,
    asset_map_revision: str,
):
    _require_evidence_kind(
        asset, DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT
    )
    evidence = canonical_digest(
        {
            "asset_id": asset.asset_id,
            "role": asset.role,
            "adapter": asset.adapter_kind,
            "expected_identity": asset.expected_identity,
        }
    )
    acknowledgement = DeviceIdentityAck(
        asset.expected_identity,
        asset.evidence_kind,
        evidence,
        asset_map_revision,
    )
    return lambda: acknowledgement


def _qcmos_identity_probe(
    device: QCMOSCamera,
    asset: InstallationAsset,
    asset_map_revision: str,
):
    _require_evidence_kind(asset, DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK)
    def probe() -> DeviceIdentityAck:
        if device._dcam is None:
            raise RuntimeError(
                "qCMOS identity probe requires an already-established DCAM connection"
            )
        from Zou_lab_control.neutral_atom.devices.drivers.dcam.dcamapi4 import (
            DCAM_IDSTR,
        )

        camera_id = str(device._dcam.dev_getstring(DCAM_IDSTR.CAMERAID) or "").strip()
        model = str(device._dcam.dev_getstring(DCAM_IDSTR.MODEL) or "").strip()
        bus = str(device._dcam.dev_getstring(DCAM_IDSTR.BUS) or "").strip()
        if not camera_id:
            raise RuntimeError("qCMOS identity readback returned no CAMERAID")
        connection_receipt = _connection_receipt(device, device._dcam)
        evidence = canonical_digest(
            {
                "camera_id": camera_id,
                "model": model,
                "bus": bus,
                "connection_receipt": connection_receipt,
            }
        )
        stable_identity = f"dcam:{camera_id}"
        _match_live_identity(asset, stable_identity)
        return DeviceIdentityAck(
            stable_identity,
            DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
            evidence,
            asset_map_revision,
        )

    return probe


def _pylon_identity_probe(
    device: PylonCamera,
    asset: InstallationAsset,
    asset_map_revision: str,
):
    _require_evidence_kind(asset, DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK)
    def probe() -> DeviceIdentityAck:
        serial, model = _pylon_live_identity(device)
        if device.serial and serial != device.serial:
            raise RuntimeError(
                f"Pylon serial readback {serial!r} differs from pinned {device.serial!r}"
            )
        connection_receipt = _connection_receipt(device, device._camera)
        stable_identity = f"pylon:{serial}"
        _match_live_identity(asset, stable_identity)
        return DeviceIdentityAck(
            stable_identity,
            DeviceIdentityEvidenceKind.HARDWARE_IDENTITY_READBACK,
            canonical_digest(
                {
                    "serial": serial,
                    "model": model,
                    "connection_receipt": connection_receipt,
                }
            ),
            asset_map_revision,
        )

    return probe


def _remote_sequencer_identity_probe(
    device: RemoteSequencer,
    asset: InstallationAsset,
    asset_map_revision: str,
):
    _require_evidence_kind(
        asset, DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT
    )
    def probe() -> DeviceIdentityAck:
        if device._conn is None or getattr(device._conn, "closed", False):
            raise RuntimeError(
                "remote sequencer identity probe requires an established RPC connection"
            )
        snapshot = dict(device.snapshot())
        evidence = canonical_digest(
            {
                "host": device.host,
                "port": device.port,
                "port_catalog_fingerprint": snapshot.get(
                    "port_catalog_fingerprint"
                ),
                "clock_hz": snapshot.get("clock_hz"),
                "connection_receipt": _connection_receipt(device, device._conn),
            }
        )
        stable_identity = (
            f"installation-endpoint:remote-sequencer:{device.host}:{device.port}"
        )
        _match_live_identity(asset, stable_identity)
        return DeviceIdentityAck(
            stable_identity,
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            evidence,
            asset_map_revision,
        )

    return probe


def _camera_is_unarmed(device, key: ResourceKey) -> None:
    state = device._recent_state()
    with state["cond"]:
        if bool(state["armed"]):
            raise RuntimeError(f"camera {key} still owns an armed acquisition session")
        if state["pending"]:
            raise RuntimeError(f"camera {key} still retains undrained pending frames")


def _connection_receipt(device: object, handle: object) -> str:
    """Return one process-local receipt for the lifetime of an exact live handle."""

    with _CONNECTION_RECEIPTS_LOCK:
        current = _CONNECTION_RECEIPTS.get(device)
        if current is not None and current[0] is handle:
            return current[1]
        receipt = uuid.uuid4().hex
        _CONNECTION_RECEIPTS[device] = (handle, receipt)
        return receipt


def _pylon_live_identity(device: PylonCamera) -> tuple[str, str]:
    """Read identity through the live GenICam node map; cached DeviceInfo is insufficient."""

    camera = device._camera
    if camera is None or getattr(device, "_connection_token", None) is None:
        raise RuntimeError(
            "Pylon identity probe requires an already-established SDK connection"
        )
    is_open = getattr(camera, "IsOpen", None)
    is_removed = getattr(camera, "IsCameraDeviceRemoved", None)
    if not callable(is_open) or not callable(is_removed):
        raise RuntimeError(
            "Pylon adapter cannot prove live connection: IsOpen/IsCameraDeviceRemoved missing"
        )
    if not bool(is_open()):
        raise RuntimeError("Pylon SDK handle is not open")
    if bool(is_removed()):
        raise RuntimeError("Pylon camera removal has been detected")
    try:
        serial_node = getattr(camera, "DeviceSerialNumber")
        model_node = getattr(camera, "DeviceModelName")
        serial = str(serial_node.GetValue() or "").strip()
        model = str(model_node.GetValue() or "").strip()
    except BaseException as exc:
        # A real node-map access is intentional: IsOpen may remain true after a cable pull,
        # and the SDK may update its removal flag only after the first failing transport access.
        try:
            removed_after_error = bool(is_removed())
        except BaseException:
            removed_after_error = True
        suffix = " (device removed)" if removed_after_error else ""
        raise RuntimeError(f"Pylon live node-map identity readback failed{suffix}") from exc
    if bool(is_removed()):
        raise RuntimeError("Pylon camera was removed during live identity readback")
    if not serial:
        raise RuntimeError("Pylon live node-map returned no serial number")
    return serial, model


def _disarm_camera(device, key: ResourceKey) -> CleanupStepAck:
    failures: list[tuple[str, BaseException]] = []
    for name, action in (("disarm", device.disarm), ("stop", device.stop)):
        try:
            action()
        except BaseException as exc:
            failures.append((name, exc))
    if failures:
        detail = "; ".join(
            f"{name}: {type(exc).__name__}: {exc}" for name, exc in failures
        )
        raise RuntimeError(f"camera {key} cleanup did not acknowledge every step: {detail}") from failures[0][1]
    return CleanupStepAck(
        SafetyOperation.DISARM,
        canonical_digest({"resource": str(key), "operation": "DISARM"}),
    )


def _qcmos_safe_contract(device: QCMOSCamera, key: ResourceKey):
    def cleanup() -> CleanupStepAck:
        device.disarm()
        return CleanupStepAck(
            SafetyOperation.DISARM,
            canonical_digest({"resource": str(key), "operation": "DISARM"}),
        )

    def verify() -> SafeStateAck:
        from Zou_lab_control.neutral_atom.devices.drivers.dcam.dcamapi4 import (
            DCAMCAP_STATUS,
        )

        _camera_is_unarmed(device, key)
        if device._dcam is None:
            raise RuntimeError(f"qCMOS {key} has no live handle for terminal readback")
        status = device._dcam.cap_status()
        if status != DCAMCAP_STATUS.READY:
            raise RuntimeError(f"qCMOS {key} capture status is not READY: {status!r}")
        return SafeStateAck(
            canonical_digest(
                {"resource": str(key), "armed": False, "dcam_status": "READY"}
            )
        )

    return cleanup, verify


def _pylon_safe_contract(device: PylonCamera, key: ResourceKey):
    def verify() -> SafeStateAck:
        _camera_is_unarmed(device, key)
        serial, model = _pylon_live_identity(device)
        if bool(device._camera.IsGrabbing()):
            raise RuntimeError(f"Pylon camera {key} still reports an active grab stream")
        # Re-read after IsGrabbing so a removal discovered by that SDK access cannot mint SAFE.
        final_serial, final_model = _pylon_live_identity(device)
        if (final_serial, final_model) != (serial, model):
            raise RuntimeError(f"Pylon camera {key} identity changed during SAFE verification")
        return SafeStateAck(
            canonical_digest(
                {
                    "resource": str(key),
                    "armed": False,
                    "sdk_grabbing": False,
                    "serial": serial,
                    "model": model,
                }
            )
        )

    return lambda: _disarm_camera(device, key), verify


def _virtual_camera_safe_contract(device, key: ResourceKey):
    def verify() -> SafeStateAck:
        _camera_is_unarmed(device, key)
        return SafeStateAck(
            canonical_digest({"resource": str(key), "armed": False})
        )

    return lambda: _disarm_camera(device, key), verify


def _virtual_sequencer_safe_contract(device: VirtualSequencer, key: ResourceKey):
    def cleanup() -> CleanupStepAck:
        device.set_safe_state()
        return CleanupStepAck(
            SafetyOperation.SAFE_STATE,
            canonical_digest({"resource": str(key), "operation": "SAFE_STATE"}),
        )

    def verify() -> SafeStateAck:
        snapshot = dict(device.snapshot())
        if snapshot.get("state") != "safe":
            raise RuntimeError(
                f"virtual sequencer {key} is not safe: {snapshot.get('state')!r}"
            )
        if device.firing is not None or device.last_fired is not None:
            raise RuntimeError(f"virtual sequencer {key} still retains an active output")
        if device._scan_info is not None or device._scan_fire_time != 0.0:
            raise RuntimeError(f"virtual sequencer {key} still retains active scan state")
        if snapshot.get("prepared_program") is not None:
            raise RuntimeError(f"virtual sequencer {key} still retains a prepared program")
        return SafeStateAck(
            canonical_digest(
                {"resource": str(key), "state": "safe", "firing": None}
            )
        )

    return cleanup, verify


def _remote_sequencer_cleanup(device: RemoteSequencer, key: ResourceKey):
    def cleanup() -> CleanupStepAck:
        device.set_safe_state()
        return CleanupStepAck(
            SafetyOperation.SAFE_STATE,
            canonical_digest({"resource": str(key), "operation": "SAFE_STATE"}),
        )

    return cleanup


def _remote_sequencer_unqualified_verifier(key: ResourceKey):
    def verify() -> SafeStateAck:
        raise RuntimeError(
            f"remote sequencer {key} has no qualified raw hardware terminal/tail "
            "readback contract in the frozen-bitstream baseline"
        )

    return verify


def _remote_sequencer_no_go(key: ResourceKey):
    def check() -> None:
        raise RuntimeError(
            f"remote sequencer {key} is NO-GO through the legacy fence until H1 "
            "qualifies its existing raw status and post-terminal output-tail evidence"
        )

    return check


def _requires_persistent_safety(
    registrations: tuple[LegacyDeviceRegistration, ...],
) -> bool:
    return any(
        registration.hazardous
        and not isinstance(
            registration.device,
            (VirtualCamera, VirtualMotCamera, VirtualSequencer),
        )
        for registration in registrations
    )


__all__ = [
    "LegacyNeutralAtomRuntime",
    "default_asset_map_path",
    "default_safety_journal_path",
    "registrations_for",
]
