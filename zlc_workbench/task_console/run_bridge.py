"""RUN seam: one console node's Run lifecycle, off the GUI thread.

Seam 2 of the composition root's rewiring contract (``app.py``).  A console node
is a frozen typed request plus a Run: the CATALOG seam freezes the request
(:mod:`.catalog_bridge`), this module owns what happens next -- prepare, start,
cancel, and the snapshot the board polls each tick.

The GUI thread never blocks on the domain.  Every prepare/start/cancel round
trip is submitted to :class:`~zlc_workbench.run_owner.QtRunOwnerMailbox`, the
same worker mailbox the Workbench's own monitor window uses, and the console
reads results by draining completions on its tick.  A node that is stopping
therefore keeps the board interactive; a node that dies reports its failure
instead of leaving a button stuck.

Layering: this module does NOT import the notebook facade.  It takes a
``prepare`` callable the composition root supplies (the same shape the Workbench
windows take), so the console package stays free of domain authority.
"""

from __future__ import annotations

from typing import Callable

from zlc_workbench.run_owner import QtRunOwnerMailbox

__all__ = ["ConsoleRunNode"]


class ConsoleRunNode:
    """The Run lifecycle of ONE console node (a logic row / camera panel).

    ``prepare`` receives the node's frozen request and returns the domain's
    prepared command.  ``view_factory``, when the node's prepared command wants
    a live view, is handed straight to ``start_with_view`` -- the MONITOR seam
    owns what that factory builds, this seam only sequences the call.
    """

    def __init__(self, spec, values, *, prepare: Callable[[object], object],
                 request_owner_wake: Callable[[], None]) -> None:
        if not callable(prepare):
            raise TypeError("prepare must be callable")
        self._spec = spec
        self._values = dict(values)
        self._prepare = prepare
        # One worker thread per node: a node's prepare/start/cancel jobs must not
        # interleave with each other, and thread affinity is what lets a domain
        # port that is not thread-safe be driven from here at all.
        self._owner = QtRunOwnerMailbox(
            request_owner_wake,
            thread_name_prefix=f"console-{spec.key.stable_definition_id}",
            max_workers=1,
        )
        self._request = spec.build_request(self._values)
        self._prepared: object | None = None
        self._handle = None
        self._snapshot = None
        self._error: str | None = None
        self._stop_requested = False

    # ----------------------------------------------------------------- facts
    @property
    def spec(self):
        return self._spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def request(self):
        """The frozen typed request -- the node's identity for this run."""

        return self._request

    @property
    def handle(self):
        return self._handle

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def running(self) -> bool:
        snapshot = self._snapshot
        if self._handle is None:
            return False
        return snapshot is None or not snapshot.state.terminal

    # ------------------------------------------------------------ lifecycle
    def start(self, start: Callable[[object], object]) -> None:
        """Prepare and start on the worker.  Returns immediately.

        ``start`` receives the prepared command and returns its RunHandle.  How a
        command starts is the command's own business -- a camera monitor only
        starts WITH a live view and its byte budget, a finite capture just starts
        -- so this seam sequences the call rather than knowing the shapes.

        One submission does both prepare and start: a prepared command that is
        never started is a reservation the operator did not ask for, and
        splitting them lets a stale prepare outlive the request it was frozen
        from.
        """

        if not callable(start):
            raise TypeError("start must be callable")
        if self.running:
            return
        self._error = None
        self._stop_requested = False
        generation = self._owner.begin_generation()
        request = self._request
        prepare = self._prepare
        self._owner.submit("start", lambda: start(prepare(request)),
                           generation=generation)

    def cancel(self, reason: str = "Console user requested stop") -> None:
        """Ask the Run to stop.  Never waits for the worker to reap it."""

        handle = self._handle
        self._stop_requested = True
        if handle is None or handle.snapshot().state.terminal:
            return
        handle.cancel(reason)

    def poll(self):
        """Drain worker completions and refresh the snapshot.  Call per tick.

        Returns the current ``RunSnapshot`` (or None before the first start).
        The board reads state from the RETURNED snapshot rather than from a
        flag this object also keeps, so the button state and the displayed
        diagnostics can never disagree about the same run.
        """

        for completion in self._owner.drain_completions():
            if completion.generation != self._owner.generation:
                continue          # a job from a superseded generation; its result is stale
            error = completion.future.exception()
            if error is not None:
                self._error = f"{type(error).__name__}: {error}"
                self._handle = None
                self._owner.mark_owner_reaped()
                continue
            if completion.kind == "start":
                self._handle = completion.future.result()
                self._owner.set_handle(self._handle)
        handle = self._handle
        if handle is not None:
            self._snapshot = handle.snapshot()
            if self._snapshot.state.terminal:
                self._owner.mark_owner_reaped()
        return self._snapshot

    def shutdown(self) -> None:
        self.cancel("Console is closing")
        self._owner.shutdown()
