"""Installation generation state and capability-free public observation.

This module contains no hardware lifecycle policy.  ``InstallationSupervisor`` is the
stable composition owner that keeps the runtime authority private while publishing a
strictly ordered ``AVAILABLE | UNAVAILABLE`` state projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
import uuid

from zlc_storage import canonical_text, positive_integer as _positive_int

from .device_catalog import (
    DeviceCatalogView,
    InstallationAvailability,
    _catalog_from_device_set,
    unavailable_catalog,
)

@dataclass(frozen=True, slots=True)
class RecoveryStatusRef:
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            canonical_text(self.value, "recovery status ref"),
        )


@dataclass(frozen=True, slots=True)
class PublicInstallationSnapshot:
    catalog: DeviceCatalogView
    availability: InstallationAvailability
    recovery_status_ref: RecoveryStatusRef | None

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, DeviceCatalogView):
            raise TypeError("public installation snapshot needs DeviceCatalogView")
        if not isinstance(self.availability, InstallationAvailability):
            raise TypeError("snapshot availability must be InstallationAvailability")
        if self.catalog.availability is not self.availability:
            raise ValueError("snapshot availability must match its catalog")
        expected = (
            None
            if self.catalog.recovery_status_ref is None
            else RecoveryStatusRef(self.catalog.recovery_status_ref)
        )
        if self.recovery_status_ref != expected:
            raise ValueError("snapshot recovery ref must match its catalog")

    @property
    def installation_state_revision(self) -> int:
        return self.catalog.installation_state_revision


@dataclass(frozen=True, slots=True)
class _AvailableInstallationState:
    catalog: DeviceCatalogView
    device_set: object

    def __post_init__(self) -> None:
        if self.catalog.availability is not InstallationAvailability.AVAILABLE:
            raise ValueError("available state needs an available catalog")


@dataclass(frozen=True, slots=True)
class _UnavailableInstallationState:
    catalog: DeviceCatalogView
    recovery_status_ref: RecoveryStatusRef | None

    def __post_init__(self) -> None:
        if self.catalog.availability is InstallationAvailability.AVAILABLE:
            raise ValueError("unavailable state cannot have an available catalog")
        expected = (
            None
            if self.catalog.recovery_status_ref is None
            else RecoveryStatusRef(self.catalog.recovery_status_ref)
        )
        if self.recovery_status_ref != expected:
            raise ValueError("unavailable state recovery ref must match its catalog")


_InstallationState = _AvailableInstallationState | _UnavailableInstallationState


@dataclass(frozen=True, slots=True)
class _SwapRecoveryContext:
    transition_token: object
    candidate_device_set: object
    prepared_binding_state: object
    reason: str


class DeviceCatalogReader:
    """Public snapshot/watch port; it never returns the private installation state."""

    __slots__ = ("_supervisor",)

    def __init__(self, supervisor: "InstallationSupervisor") -> None:
        self._supervisor = supervisor

    def snapshot(self) -> PublicInstallationSnapshot:
        return self._supervisor.snapshot_public()

    def watch(
        self, after_revision: int, *, timeout: float | None = None
    ) -> PublicInstallationSnapshot:
        return self._supervisor.watch_public(after_revision, timeout=timeout)


class InstallationSupervisor:
    """Stable owner of one runtime authority and its generation state pointer."""

    __slots__ = (
        "_condition",
        "_installation_id",
        "_runtime_authority",
        "_state",
        "_next_installation_generation",
        "_next_state_revision",
        "_next_catalog_revision",
        "_catalog_reader",
        "_retained_raw_graphs",
        "_recovery_context",
    )

    def __init__(
        self,
        device_set: object,
        runtime_authority: object,
        *,
        installation_id: str | None = None,
    ) -> None:
        normalized_id = str(installation_id or uuid.uuid4().hex)
        if not normalized_id or normalized_id.strip() != normalized_id:
            raise ValueError("installation id must be canonical non-empty text")
        self._condition = threading.Condition(threading.RLock())
        self._installation_id = normalized_id
        self._runtime_authority = runtime_authority
        self._next_installation_generation = 1
        self._next_state_revision = 1
        self._next_catalog_revision = 1
        catalog = _catalog_from_device_set(
            device_set,
            installation_id=normalized_id,
            installation_generation=1,
            installation_state_revision=1,
            revision=1,
        )
        self._state: _InstallationState = _AvailableInstallationState(
            catalog, device_set
        )
        self._catalog_reader = DeviceCatalogReader(self)
        self._retained_raw_graphs: tuple[object, ...] = ()
        self._recovery_context: object | None = None

    @property
    def catalog_reader(self) -> DeviceCatalogReader:
        return self._catalog_reader

    @property
    def catalog(self) -> DeviceCatalogView:
        with self._condition:
            return self._state.catalog

    def snapshot_public(self) -> PublicInstallationSnapshot:
        with self._condition:
            return self._public_snapshot_locked()

    def watch_public(
        self, after_revision: int, *, timeout: float | None = None
    ) -> PublicInstallationSnapshot:
        after_revision = _positive_int(after_revision, "after revision")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError("watch timeout must be finite and non-negative")
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("watch timeout must be finite and non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            current = self._state.catalog.installation_state_revision
            if after_revision > current:
                raise ValueError("after revision is newer than the installation")
            while self._state.catalog.installation_state_revision <= after_revision:
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("installation state did not advance")
                    self._condition.wait(remaining)
            return self._public_snapshot_locked()

    def _available_device_set(self):
        with self._condition:
            if not isinstance(self._state, _AvailableInstallationState):
                raise RuntimeError(
                    f"installation is {self._state.catalog.availability.value}"
                )
            return self._state.device_set

    def _runtime(self):
        return self._runtime_authority

    def _retain_recovery_context(self, context: object) -> None:
        if context is None:
            raise TypeError("recovery context cannot be None")
        with self._condition:
            if isinstance(self._state, _AvailableInstallationState):
                raise RuntimeError("an available installation cannot retain swap recovery")
            if self._recovery_context is not None:
                raise RuntimeError("installation already owns swap recovery context")
            self._recovery_context = context

    def _retain_swap_recovery(
        self,
        *,
        transition_token: object,
        candidate_device_set: object,
        prepared_binding_state: object,
        reason: str,
    ) -> None:
        normalized_reason = str(reason)
        if not normalized_reason:
            raise ValueError("swap recovery reason cannot be empty")
        context = _SwapRecoveryContext(
            transition_token,
            candidate_device_set,
            prepared_binding_state,
            normalized_reason,
        )
        with self._condition:
            if isinstance(self._state, _AvailableInstallationState):
                raise RuntimeError("an available installation cannot retain swap recovery")
            if self._recovery_context is not None:
                raise RuntimeError("installation already owns swap recovery context")
            self._recovery_context = context
            if all(
                candidate_device_set is not item
                for item in self._retained_raw_graphs
            ):
                self._retained_raw_graphs = (
                    *self._retained_raw_graphs,
                    candidate_device_set,
                )

    def _recovery(self) -> object | None:
        with self._condition:
            return self._recovery_context

    def _device_sets_for_shutdown(self) -> tuple[object, ...]:
        with self._condition:
            values = list(self._retained_raw_graphs)
            if isinstance(self._state, _AvailableInstallationState):
                values.append(self._state.device_set)
            unique: list[object] = []
            seen: set[int] = set()
            for value in values:
                if id(value) not in seen:
                    seen.add(id(value))
                    unique.append(value)
            return tuple(unique)

    def _publish_swapping(self) -> PublicInstallationSnapshot:
        with self._condition:
            if not isinstance(self._state, _AvailableInstallationState):
                raise RuntimeError("only an available installation can begin swapping")
            generation, state_revision, catalog_revision = self._advance_locked()
            catalog = unavailable_catalog(
                self._state.catalog,
                installation_generation=generation,
                installation_state_revision=state_revision,
                revision=catalog_revision,
                availability=InstallationAvailability.SWAPPING,
            )
            self._retained_raw_graphs = (self._state.device_set,)
            self._state = _UnavailableInstallationState(catalog, None)
            self._condition.notify_all()
            return self._public_snapshot_locked()

    def _publish_recovery_required(
        self, recovery_status_ref: RecoveryStatusRef
    ) -> PublicInstallationSnapshot:
        if not isinstance(recovery_status_ref, RecoveryStatusRef):
            raise TypeError("recovery status ref must be RecoveryStatusRef")
        with self._condition:
            if isinstance(self._state, _AvailableInstallationState):
                raise RuntimeError(
                    "recovery-required cannot replace a still-available installation"
                )
            generation, state_revision, catalog_revision = self._advance_locked()
            catalog = unavailable_catalog(
                self._state.catalog,
                installation_generation=generation,
                installation_state_revision=state_revision,
                revision=catalog_revision,
                availability=InstallationAvailability.RECOVERY_REQUIRED,
                recovery_status_ref=recovery_status_ref.value,
            )
            self._state = _UnavailableInstallationState(
                catalog, recovery_status_ref
            )
            self._condition.notify_all()
            return self._public_snapshot_locked()

    def _publish_available(self, device_set: object) -> PublicInstallationSnapshot:
        with self._condition:
            generation, state_revision, catalog_revision = self._advance_locked()
            catalog = _catalog_from_device_set(
                device_set,
                installation_id=self._installation_id,
                installation_generation=generation,
                installation_state_revision=state_revision,
                revision=catalog_revision,
            )
            self._state = _AvailableInstallationState(catalog, device_set)
            self._retained_raw_graphs = ()
            self._recovery_context = None
            self._condition.notify_all()
            return self._public_snapshot_locked()

    def _advance_locked(self) -> tuple[int, int, int]:
        self._next_installation_generation += 1
        self._next_state_revision += 1
        self._next_catalog_revision += 1
        return (
            self._next_installation_generation,
            self._next_state_revision,
            self._next_catalog_revision,
        )

    def _public_snapshot_locked(self) -> PublicInstallationSnapshot:
        state = self._state
        recovery_ref = (
            state.recovery_status_ref
            if isinstance(state, _UnavailableInstallationState)
            else None
        )
        return PublicInstallationSnapshot(
            state.catalog,
            state.catalog.availability,
            recovery_ref,
        )


__all__ = [
    "DeviceCatalogReader",
    "InstallationSupervisor",
    "PublicInstallationSnapshot",
    "RecoveryStatusRef",
]
