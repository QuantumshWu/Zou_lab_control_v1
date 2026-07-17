"""Composition-only worker mailbox and Run ownership for Qt surfaces."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
from typing import Callable, NamedTuple

from zlc_neutral_atom.runtime.run import RunHandle, RunSnapshot


class OwnerCompletion(NamedTuple):
    kind: str
    generation: int
    future: Future


class QtRunOwnerMailbox:
    """Own mechanics shared by Run windows, never their product policy."""

    def __init__(
        self,
        request_owner_wake: Callable[[], None],
        *,
        thread_name_prefix: str,
        max_workers: int = 2,
    ) -> None:
        self._wake = request_owner_wake
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._worker_thread_affine = max_workers == 1
        self._lock = threading.Lock()
        self._tracked: set[Future] = set()
        self._completions: list[OwnerCompletion] = []
        self._attachments: list[tuple[int, object]] = []
        self._pending_snapshot: tuple[int, RunSnapshot] | None = None
        self._generation = 0
        self._handle: RunHandle | None = None
        self._owner_reaped = True
        self._terminal_job_inflight = False

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def handle(self) -> RunHandle | None:
        return self._handle

    @property
    def owner_reaped(self) -> bool:
        return self._owner_reaped

    @property
    def worker_idle(self) -> bool:
        with self._lock:
            return not self._tracked and not self._completions

    @property
    def worker_thread_affine(self) -> bool:
        """Whether every submitted job is guaranteed one stable OS thread."""

        return self._worker_thread_affine

    @property
    def has_pending_owner_work(self) -> bool:
        with self._lock:
            return bool(self._tracked or self._completions or self._attachments)

    def begin_generation(self) -> int:
        self._generation += 1
        self._handle = None
        self._owner_reaped = False
        self._terminal_job_inflight = False
        with self._lock:
            self._pending_snapshot = None
        return self._generation

    def set_handle(self, handle: RunHandle) -> None:
        self._handle = handle

    def mark_owner_reaped(self) -> None:
        self._owner_reaped = True

    def submit(
        self,
        kind: str,
        work: Callable[[], object],
        *,
        generation: int | None = None,
    ) -> Future:
        generation = self._generation if generation is None else generation
        future = self._pool.submit(work)
        with self._lock:
            self._tracked.add(future)

        def done(completed: Future) -> None:
            with self._lock:
                self._tracked.discard(completed)
                self._completions.append(
                    OwnerCompletion(kind, generation, completed)
                )
            self._wake()

        future.add_done_callback(done)
        return future

    def submit_render(self, work: Callable[[], object]) -> Future:
        return self.submit("render", work)

    def drain_completions(self) -> tuple[OwnerCompletion, ...]:
        with self._lock:
            pending = tuple(self._completions)
            self._completions.clear()
        return pending

    def post_attachment(self, payload: object, *, generation: int) -> None:
        with self._lock:
            self._attachments.append((generation, payload))
        self._wake()

    def drain_attachments(self) -> tuple[tuple[int, object], ...]:
        with self._lock:
            pending = tuple(self._attachments)
            self._attachments.clear()
        return pending

    def enqueue_snapshot(
        self,
        snapshot: RunSnapshot,
        *,
        generation: int | None = None,
        last_snapshot: RunSnapshot | None = None,
    ) -> bool:
        generation = self._generation if generation is None else generation
        with self._lock:
            pending = self._pending_snapshot
            if snapshot == last_snapshot or (
                pending is not None and pending[1] == snapshot
            ):
                return False
            self._pending_snapshot = (generation, snapshot)
        return True

    def poll_snapshot(self, last_snapshot: RunSnapshot | None) -> bool:
        handle = self._handle
        return handle is not None and self.enqueue_snapshot(
            handle.snapshot(), last_snapshot=last_snapshot
        )

    def take_pending_snapshot(self) -> tuple[int, RunSnapshot] | None:
        with self._lock:
            pending, self._pending_snapshot = self._pending_snapshot, None
        return pending

    def begin_terminal_job(
        self,
        kind: str,
        work: Callable[[], object],
    ) -> bool:
        if self._terminal_job_inflight or self._owner_reaped:
            return False
        self._terminal_job_inflight = True
        try:
            self.submit(kind, work)
        except BaseException:
            self._terminal_job_inflight = False
            raise
        return True

    def finish_terminal_job(
        self,
        generation: int,
        *,
        owner_reaped: bool,
    ) -> None:
        if generation != self._generation:
            return
        self._terminal_job_inflight = False
        self._owner_reaped = self._owner_reaped or owner_reaped

    def shutdown(self) -> None:
        if self.has_pending_owner_work:
            raise RuntimeError("cannot close Run owner with pending work")
        self._pool.shutdown(wait=False)


__all__ = ["OwnerCompletion", "QtRunOwnerMailbox"]
