"""Application-owned Experiment service graph and lifecycle borrows.

This module contains only composition mechanics shared by application facades.
Concrete Logic-node behavior and resources belong to each node-local API.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from zlc_data import FitCancelled
from zlc_neutral_atom.artifacts import FitResultRepository
from zlc_neutral_atom.capture.artifact import CaptureRepository
from zlc_neutral_atom.devices.sequencer.application import PulseApplicationOwner
from zlc_neutral_atom.installation import DeviceCatalogView
from zlc_neutral_atom.installation_config import InstallationConfigDocument


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """Application-owned roots for user-authored and generated files."""

    pulses_root: Path
    tasks_root: Path
    output_root: Path
    repository_root: Path

    def __post_init__(self) -> None:
        for name in (
            "pulses_root",
            "tasks_root",
            "output_root",
            "repository_root",
        ):
            value = Path(getattr(self, name)).expanduser()
            if not value.is_absolute():
                raise ValueError(f"WorkspacePaths.{name} must be absolute")
            object.__setattr__(self, name, value.resolve())

    @classmethod
    def for_workspace(
        cls,
        authored_root: str | Path,
        *,
        repository_root: str | Path,
    ) -> "WorkspacePaths":
        """Build the conventional four roots from two explicit authorities."""

        authored = Path(authored_root).expanduser()
        repository = Path(repository_root).expanduser()
        if not authored.is_absolute() or not repository.is_absolute():
            raise ValueError("workspace and repository roots must be absolute")
        authored = authored.resolve()
        repository = repository.resolve()
        return cls(
            authored / "pulses",
            authored / "tasks",
            repository / "output",
            repository,
        )

@runtime_checkable
class WorkbenchHandle(Protocol):
    """Lifecycle port retained by the application that owns a GUI borrow."""

    @property
    def permanently_closed(self) -> bool: ...

    def restore_window(self) -> None: ...

    def request_owner_close(self) -> None: ...

    def wait_owner_closed(self, timeout: float) -> bool: ...


@dataclass
class ExperimentCloseAttempt:
    """One close owner and the completion fact shared by all concurrent callers."""

    owner_thread_id: int
    completed: threading.Event = field(default_factory=threading.Event)
    failure: BaseException | None = None


def wait_for_close_attempt(
    attempt: ExperimentCloseAttempt,
    handles: tuple[WorkbenchHandle, ...],
) -> None:
    """Wait for one close owner without starving a caller-owned GUI loop.

    The Workbench handle is the only application port that knows whether its
    caller is the Qt owner.  Short waits therefore let that implementation pump
    the existing owner event loop while foreign callers simply wait on its
    terminal event.  This caller never performs teardown or publishes a second
    outcome; only ``attempt.owner_thread_id`` owns those transitions.
    """

    if not isinstance(attempt, ExperimentCloseAttempt):
        raise TypeError("attempt must be ExperimentCloseAttempt")
    if not handles:
        attempt.completed.wait()
        return
    while not attempt.completed.is_set():
        for handle in handles:
            if attempt.completed.is_set():
                return
            try:
                handle.wait_owner_closed(0.05)
            except Exception:
                # The attempt owner performs the authoritative wait and
                # publishes its exact failure.  A concurrent waiter only keeps
                # its own GUI owner responsive and must not invent a second
                # close result.
                pass
        # Already-closed handles return immediately; yield briefly while the
        # sole owner finishes repository teardown.
        attempt.completed.wait(0.005)


@dataclass
class ExperimentServices:
    workspace_paths: WorkspacePaths
    installation: object
    runtime: object
    capture_repository: CaptureRepository
    fit_repository: FitResultRepository
    catalog: DeviceCatalogView
    installation_config: InstallationConfigDocument
    pulse_application: PulseApplicationOwner
    operation_lock: threading.RLock
    fit_operations_drained: threading.Event
    fit_operation_thread_counts: dict[int, int]
    gui_handles: dict[str, WorkbenchHandle]
    active_fit_operations: int = 0
    state: str = "OPEN"
    closing_gui_handles: tuple[WorkbenchHandle, ...] = ()
    close_attempt: ExperimentCloseAttempt | None = None


@contextmanager
def service_guard(services: ExperimentServices) -> Iterator[ExperimentServices]:
    """Borrow the one Experiment-owned service graph while it is open."""

    if not isinstance(services, ExperimentServices):
        raise TypeError("services must be ExperimentServices")
    with services.operation_lock:
        if services.state != "OPEN":
            raise RuntimeError("Experiment is closing or closed")
        yield services


@contextmanager
def fit_service_guard(
    services: ExperimentServices,
) -> Iterator[ExperimentServices]:
    """Keep repositories alive for one long Fit without serializing figures."""

    if not isinstance(services, ExperimentServices):
        raise TypeError("services must be ExperimentServices")
    with services.operation_lock:
        if services.state != "OPEN":
            raise RuntimeError("Experiment is closing or closed")
        if services.active_fit_operations == 0:
            services.fit_operations_drained.clear()
        services.active_fit_operations += 1
        thread_id = threading.get_ident()
        services.fit_operation_thread_counts[thread_id] = (
            services.fit_operation_thread_counts.get(thread_id, 0) + 1
        )
    completed = False
    try:
        yield services
        completed = True
    finally:
        with services.operation_lock:
            if services.active_fit_operations <= 0:
                raise RuntimeError("Experiment Fit operation count underflow")
            services.active_fit_operations -= 1
            thread_count = services.fit_operation_thread_counts.get(thread_id, 0)
            if thread_count <= 0:
                raise RuntimeError("Experiment Fit thread count underflow")
            if thread_count == 1:
                services.fit_operation_thread_counts.pop(thread_id)
            else:
                services.fit_operation_thread_counts[thread_id] = thread_count - 1
            if services.active_fit_operations == 0:
                services.fit_operations_drained.set()
            remained_open = services.state == "OPEN"
    if completed and not remained_open:
        raise FitCancelled("Experiment began closing during Fit execution")


def resolve_role(
    catalog: DeviceCatalogView,
    requested: str | None,
    domain: str,
    preferred: tuple[str, ...],
) -> str:
    """Resolve one typed installation role without exposing the runtime graph."""

    if requested is not None:
        info = catalog.require(requested)
        if info.domain != domain:
            raise ValueError(
                f"device role {requested!r} is {info.domain!r}, not {domain!r}"
            )
        return requested
    for role in preferred:
        info = catalog.find(role)
        if info is not None and info.domain == domain:
            return role
    candidates = catalog.roles(domain)
    if len(candidates) != 1:
        raise ValueError(
            f"installation has {len(candidates)} {domain} roles; choose one explicitly"
        )
    return candidates[0]


__all__ = [
    "ExperimentCloseAttempt",
    "ExperimentServices",
    "WorkspacePaths",
    "WorkbenchHandle",
    "fit_service_guard",
    "resolve_role",
    "service_guard",
    "wait_for_close_attempt",
]
