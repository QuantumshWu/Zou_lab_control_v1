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

from zlc_data import StreamGenerationId
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
    """Application-owned roots derived from one explicit project root."""

    project_root: Path
    pulses_root: Path
    tasks_root: Path
    output_root: Path

    def __post_init__(self) -> None:
        for name in (
            "project_root",
            "pulses_root",
            "tasks_root",
            "output_root",
        ):
            value = Path(getattr(self, name)).expanduser()
            if not value.is_absolute():
                raise ValueError(f"WorkspacePaths.{name} must be absolute")
            object.__setattr__(self, name, value.resolve())

    @classmethod
    def for_workspace(
        cls,
        project_root: str | Path,
    ) -> "WorkspacePaths":
        """Build the conventional roots from one explicit project authority."""

        project = Path(project_root).expanduser()
        if not project.is_absolute():
            raise ValueError("project root must be absolute")
        project = project.resolve()
        return cls(
            project,
            project / "pulses",
            project / "tasks",
            project / "_output",
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


_SignalGenerationRef = tuple[str, StreamGenerationId]


@dataclass(frozen=True, slots=True)
class _ActiveRun:
    """The Experiment's sole control-plane record for one admitted Run."""

    handle: RunHandle
    preemptible: bool
    signal_generation_ref: _SignalGenerationRef | None
    owner: object | None


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
        # sole owner finishes service teardown.
        attempt.completed.wait(0.005)


@dataclass
class ExperimentServices:
    workspace_paths: WorkspacePaths
    installation: object
    runtime: object
    captures_root: Path
    calibrations_root: Path
    catalog: DeviceCatalogView
    installation_config: InstallationConfigDocument
    pulse_application: PulseApplicationOwner
    signal_plane: SignalDataPlane
    operation_lock: threading.RLock
    admission_lock: threading.RLock
    operations_drained: threading.Event
    operation_thread_counts: dict[int, int]
    active_runs: dict[str, _ActiveRun]
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
        for run_id, active in services.active_runs.items()
        if active.handle.snapshot().state.terminal
    )
    for run_id in terminal:
        active = services.active_runs.get(run_id)
        if active is None:
            continue
        if active.signal_generation_ref is not None:
            services.signal_plane.release_generation_source(
                active.signal_generation_ref
            )
        services.active_runs.pop(run_id, None)


def _owner_signal_generation_ref(
    owner: object | None,
) -> _SignalGenerationRef | None:
    if owner is None:
        return None
    reference = getattr(owner, "signal_generation_ref", None)
    if reference is None:
        return None
    if (
        not isinstance(reference, tuple)
        or len(reference) != 2
        or not isinstance(reference[0], str)
        or not reference[0].strip()
        or not isinstance(reference[1], StreamGenerationId)
    ):
        raise TypeError(
            "Run application owner exposes an invalid signal_generation_ref"
        )
    return reference


def _start_frozen_plan(
    services: ExperimentServices,
    plan: RunPlan,
    cancel_requested: Callable[[], bool] | None,
    signal_generation_ref: _SignalGenerationRef | None,
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
            if signal_generation_ref is not None:
                services.signal_plane.require_active_generation(
                    signal_generation_ref
                )
            return services.runtime.start(plan)

    owner = plan.lifecycle_owner
    gate = None if owner is None else getattr(owner, "start_if_not_cancelled", None)
    handle = start_runtime() if not callable(gate) else gate(start_runtime)
    if not isinstance(handle, RunHandle):
        raise TypeError("Run starter returned a non-RunHandle")
    return handle


def _record_admitted_run(
    services: ExperimentServices,
    handle: RunHandle,
    *,
    owner: object | None,
    preemptible: bool,
    signal_generation_ref: _SignalGenerationRef | None,
) -> RunHandle:
    """Record one admitted handle in the sole application control plane."""

    run_id = handle.run_id.value
    services.active_runs[run_id] = _ActiveRun(
        handle=handle,
        preemptible=preemptible,
        signal_generation_ref=signal_generation_ref,
        owner=owner,
    )
    return handle


def _unique_generation_refs(
    references,
) -> tuple[_SignalGenerationRef, ...]:
    return tuple(dict.fromkeys(
        reference for reference in references if reference is not None
    ))


def _retirement_for_blockers_locked(
    services: ExperimentServices,
    rejection: RunStartRejected,
) -> tuple[
    tuple[str, ...],
    tuple[_ActiveRun, ...],
    tuple[_SignalGenerationRef, ...],
] | None:
    """Withdraw one complete application-authorized dependency closure."""

    blocker_run_ids = tuple(dict.fromkeys(
        blocker.conflicting_run_id for blocker in rejection.blockers
    ))
    blockers = []
    for run_id in blocker_run_ids:
        active = services.active_runs.get(run_id)
        if active is None or not active.preemptible:
            return None
        blockers.append(active)

    roots = _unique_generation_refs(
        active.signal_generation_ref for active in blockers
    )
    withdrawable = _unique_generation_refs(
        active.signal_generation_ref
        for active in services.active_runs.values()
        if active.preemptible
    )
    retired_refs: tuple[_SignalGenerationRef, ...] = ()
    if roots:
        withdrawn = services.signal_plane.withdraw_dependency_closure(
            roots,
            withdrawable,
        )
        if withdrawn is None:
            return None
        retired_refs = tuple(withdrawn)

    retired_ref_set = frozenset(retired_refs)
    blocker_set = frozenset(blocker_run_ids)
    retired_run_ids = tuple(
        run_id
        for run_id, active in services.active_runs.items()
        if run_id in blocker_set
        or active.signal_generation_ref in retired_ref_set
    )
    retired = tuple(services.active_runs[run_id] for run_id in retired_run_ids)
    if any(not active.preemptible for active in retired):
        raise RuntimeError(
            "SignalDataPlane retired a non-preemptible application Run"
        )
    for run_id in retired_run_ids:
        services.active_runs.pop(run_id)
    return retired_run_ids, retired, retired_refs


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
    owner = plan.lifecycle_owner
    signal_generation_ref = _owner_signal_generation_ref(owner)
    if cancel_requested is None and owner is not None:
        owner_cancel_requested = getattr(owner, "cancel_requested", None)
        if callable(owner_cancel_requested):
            cancel_requested = owner_cancel_requested

    with application_operation_guard(services):
        with services.admission_lock:
            _prune_terminal_runs_locked(services)
            try:
                handle = _start_frozen_plan(
                    services,
                    plan,
                    cancel_requested,
                    signal_generation_ref,
                )
            except RunStartRejected as rejection:
                retirement = _retirement_for_blockers_locked(
                    services,
                    rejection,
                )
                if retirement is None:
                    raise
                retired_run_ids, retired, retired_refs = retirement
            else:
                return _record_admitted_run(
                    services,
                    handle,
                    owner=owner,
                    preemptible=plan.preemptible,
                    signal_generation_ref=signal_generation_ref,
                )

        for active in retired:
            active.handle.cancel("replaced by a newly admitted Experiment Run")

        while not services.runtime.wait_until_released(
            retired_run_ids,
            timeout=0.05,
        ):
            # Once retirement starts, the old hardware must reach SAFE and
            # release its exact leases even if the incoming request is
            # cancelled.  Recheck that request only after safe cleanup.
            pass

        retirement_errors = (
            services.signal_plane.finish_dependency_retirement(retired_refs)
            if retired_refs
            else ()
        )
        if retirement_errors:
            raise BaseExceptionGroup(
                "signal dependency closure cleanup failed",
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
                cancel_requested,
                signal_generation_ref,
            )
            return _record_admitted_run(
                services,
                handle,
                owner=owner,
                preemptible=plan.preemptible,
                signal_generation_ref=signal_generation_ref,
            )


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
    "open_workbench_handle",
    "resolve_role",
    "service_guard",
    "wait_for_close_attempt",
]
