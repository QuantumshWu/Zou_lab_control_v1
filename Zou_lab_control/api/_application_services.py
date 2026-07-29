"""Application-owned Experiment service graph and lifecycle borrows.

This module contains only composition mechanics shared by application facades.
Concrete Logic-node behavior and resources belong to each node-local API.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol, runtime_checkable

from zlc_data import FitCancelled
from zlc_neutral_atom.artifacts import FitResultRepository
from zlc_neutral_atom.capture.artifact import CaptureRepository
from zlc_neutral_atom.devices.sequencer.application import PulseApplicationOwner
from zlc_neutral_atom.installation import DeviceCatalogView
from zlc_neutral_atom.installation_config import InstallationConfigDocument
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
from zlc_neutral_atom.runtime.cancellation import CancellationRequested
from zlc_neutral_atom.runtime.run import (
    RunHandle,
    RunPlan,
    RunStartRejected,
)


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
    signal_plane: SignalDataPlane
    operation_lock: threading.RLock
    admission_lock: threading.RLock
    operations_drained: threading.Event
    operation_thread_counts: dict[int, int]
    active_runs: dict[str, RunHandle]
    gui_handles: dict[str, WorkbenchHandle]
    active_operations: int = 0
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
def application_operation_guard(
    services: ExperimentServices,
) -> Iterator[ExperimentServices]:
    """Keep Experiment services alive without serializing a long operation."""

    if not isinstance(services, ExperimentServices):
        raise TypeError("services must be ExperimentServices")
    with services.operation_lock:
        if services.state != "OPEN":
            raise RuntimeError("Experiment is closing or closed")
        if services.active_operations == 0:
            services.operations_drained.clear()
        services.active_operations += 1
        thread_id = threading.get_ident()
        services.operation_thread_counts[thread_id] = (
            services.operation_thread_counts.get(thread_id, 0) + 1
        )
    try:
        yield services
    finally:
        with services.operation_lock:
            if services.active_operations <= 0:
                raise RuntimeError("Experiment operation count underflow")
            services.active_operations -= 1
            thread_count = services.operation_thread_counts.get(thread_id, 0)
            if thread_count <= 0:
                raise RuntimeError("Experiment operation thread count underflow")
            if thread_count == 1:
                services.operation_thread_counts.pop(thread_id)
            else:
                services.operation_thread_counts[thread_id] = thread_count - 1
            if services.active_operations == 0:
                services.operations_drained.set()


@contextmanager
def fit_service_guard(
    services: ExperimentServices,
) -> Iterator[ExperimentServices]:
    """Keep repositories alive for one long Fit without serializing figures."""

    completed = False
    with application_operation_guard(services):
        try:
            yield services
            completed = True
        finally:
            with services.operation_lock:
                remained_open = services.state == "OPEN"
    if completed and not remained_open:
        raise FitCancelled("Experiment began closing during Fit execution")


def open_workbench_handle(
    services: ExperimentServices,
    key: str | None,
    compose: Callable[[], object],
    *,
    existing_error: str | None = None,
):
    """Register or restore one application-owned GUI lifecycle borrow."""

    if not isinstance(services, ExperimentServices):
        raise TypeError("services must be ExperimentServices")
    if key is not None and (
        not isinstance(key, str) or not key.strip() or key.strip() != key
    ):
        raise ValueError("workbench key must be canonical text or None")
    if not callable(compose):
        raise TypeError("compose must be callable")
    if existing_error is not None and (
        not isinstance(existing_error, str) or not existing_error.strip()
    ):
        raise ValueError("existing_error must be non-empty text or None")
    with service_guard(services):
        existing = None if key is None else services.gui_handles.get(key)
        if existing is not None and existing.permanently_closed:
            services.gui_handles.pop(key)
            existing = None
        if existing is not None:
            if existing_error is not None:
                raise RuntimeError(existing_error)
            existing.restore_window()
            return existing
        body = compose()
        if not isinstance(body, WorkbenchHandle):
            raise TypeError(
                "Workbench composition did not return the application GUI handle port"
            )
        services.gui_handles[
            f"@window/{id(body):x}" if key is None else key
        ] = body
        return body


def _check_start_continuation(
    services: ExperimentServices,
    cancel_requested: Callable[[], bool] | None,
) -> None:
    if cancel_requested is not None and cancel_requested():
        raise CancellationRequested("Run start was cancelled before admission")
    with services.operation_lock:
        if services.state != "OPEN":
            raise CancellationRequested("Experiment began closing before admission")


def _prune_terminal_runs_locked(services: ExperimentServices) -> None:
    terminal = tuple(
        run_id
        for run_id, handle in services.active_runs.items()
        if handle.snapshot().state.terminal
    )
    for run_id in terminal:
        services.active_runs.pop(run_id, None)
        services.signal_plane.finish_run_lifecycle(run_id)


def _abort_pending_lifecycle(
    services: ExperimentServices,
    reference: tuple[str, int] | None,
) -> None:
    if reference is None:
        return
    services.signal_plane.abort_run_lifecycle(reference)


def _start_frozen_plan(
    services: ExperimentServices,
    plan: RunPlan,
    reference: tuple[str, int] | None,
    cancel_requested: Callable[[], bool] | None,
) -> RunHandle:
    """Perform the one short runtime admission at the cancellation boundary."""

    def start_runtime() -> RunHandle:
        # Closing and runtime admission share this short lock boundary.  Close
        # either wins first and prevents the Run, or waits for the admitted
        # application operation to finish; it cannot slip between the final
        # OPEN check and RunController.start().
        with services.operation_lock:
            if services.state != "OPEN":
                raise CancellationRequested(
                    "Experiment began closing before admission"
                )
            if cancel_requested is not None and cancel_requested():
                raise CancellationRequested(
                    "Run start was cancelled before admission"
                )
            return services.runtime.start(plan)

    if reference is None:
        return start_runtime()
    handle = services.signal_plane.start_run_lifecycle(reference, start_runtime)
    if not isinstance(handle, RunHandle):
        raise TypeError("Run lifecycle starter returned a non-RunHandle")
    return handle


def _record_admitted_run(
    services: ExperimentServices,
    handle: RunHandle,
    lifecycle_ref: tuple[str, int] | None,
    *,
    preemptible: bool,
) -> RunHandle:
    """Bind one admitted handle to its exact application generation."""

    run_id = handle.run_id.value
    if lifecycle_ref is not None:
        services.signal_plane.bind_run_lifecycle(
            lifecycle_ref,
            run_id,
            preemptible=preemptible,
        )
    services.active_runs[run_id] = handle
    return handle


def application_start_run(
    services: ExperimentServices,
    plan: RunPlan,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> RunHandle:
    """Admit one frozen plan, retiring one proven preemptible closure at most."""

    if not isinstance(services, ExperimentServices):
        raise TypeError("services must be ExperimentServices")
    if not isinstance(plan, RunPlan):
        raise TypeError("plan must be RunPlan")
    if plan.resource_claims and plan.lifecycle_owner is None:
        raise ValueError(
            "a hardware RunPlan must freeze its application lifecycle owner"
        )
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None")
    if cancel_requested is None and plan.lifecycle_owner is not None:
        cancel_requested = lambda: services.signal_plane.lifecycle_cancel_requested(
            plan.lifecycle_owner
        )

    lifecycle_ref: tuple[str, int] | None = None
    with application_operation_guard(services):
        try:
            with services.admission_lock:
                _prune_terminal_runs_locked(services)
                if plan.lifecycle_owner is not None:
                    lifecycle_ref = services.signal_plane.begin_run_lifecycle(
                        plan.lifecycle_owner,
                    )
                try:
                    handle = _start_frozen_plan(
                        services,
                        plan,
                        lifecycle_ref,
                        cancel_requested,
                    )
                except RunStartRejected as rejection:
                    blocker_run_ids = tuple(
                        dict.fromkeys(
                            blocker.conflicting_run_id
                            for blocker in rejection.blockers
                        )
                    )
                    retirement = (
                        services.signal_plane.retire_preemptible_run_closure(
                            blocker_run_ids
                        )
                    )
                    if retirement is None:
                        _abort_pending_lifecycle(services, lifecycle_ref)
                        lifecycle_ref = None
                        raise
                    retired_run_ids = retirement
                    retired_handles = tuple(
                        services.active_runs[run_id]
                        for run_id in retired_run_ids
                    )
                    for run_id in retired_run_ids:
                        services.active_runs.pop(run_id)
                else:
                    result = _record_admitted_run(
                        services,
                        handle,
                        lifecycle_ref,
                        preemptible=plan.preemptible,
                    )
                    lifecycle_ref = None
                    return result

            for retired in retired_handles:
                retired.cancel("replaced by a newly admitted Experiment Run")

            while not services.runtime.wait_until_released(
                retired_run_ids,
                timeout=0.05,
            ):
                # Once retirement starts, the old hardware must reach SAFE and
                # release its exact leases even if the incoming request is
                # cancelled.  Recheck that request only after safe cleanup.
                pass

            retirement_errors = (
                services.signal_plane.finish_preemptible_run_retirement(
                    retired_run_ids
                )
            )
            if retirement_errors:
                raise BaseExceptionGroup(
                    "preemptible signal closure cleanup failed",
                    list(retirement_errors),
                )
            _check_start_continuation(services, cancel_requested)

            with services.admission_lock:
                _prune_terminal_runs_locked(services)
                # A new racer is reported once.  It never starts a second
                # retirement pass and the frozen plan is never rebuilt.
                handle = _start_frozen_plan(
                    services,
                    plan,
                    lifecycle_ref,
                    cancel_requested,
                )
                result = _record_admitted_run(
                    services,
                    handle,
                    lifecycle_ref,
                    preemptible=plan.preemptible,
                )
                lifecycle_ref = None
                return result
        finally:
            if lifecycle_ref is not None:
                with services.admission_lock:
                    _abort_pending_lifecycle(services, lifecycle_ref)


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
    "application_operation_guard",
    "application_start_run",
    "fit_service_guard",
    "open_workbench_handle",
    "resolve_role",
    "service_guard",
    "wait_for_close_attempt",
]
