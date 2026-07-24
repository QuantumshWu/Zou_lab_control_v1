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

from concurrent.futures import Future
import threading
from typing import Callable

from zlc_data.console_records import console_signal_key
from zlc_workbench.run_owner import QtRunOwnerMailbox

__all__ = ["ConsoleRunNode"]


_UNRESOLVED_FINAL = object()
_BUILD_REQUEST = object()


class _StartSuppressed(Exception):
    """The user stopped this node before its starter received hardware authority."""


class ConsoleRunNode:
    """The Run lifecycle of ONE console node (a logic row / camera panel).

    ``prepare`` receives the node's frozen request and returns the domain's
    prepared command.  ``view_factory``, when the node's prepared command wants
    a live view, is handed straight to ``start_with_view`` -- the MONITOR seam
    owns what that factory builds, this seam only sequences the call.
    """

    def __init__(
        self,
        spec,
        values,
        *,
        prepare: Callable[[object], object],
        request_owner_wake: Callable[[], None],
        frozen_request: object = _BUILD_REQUEST,
    ) -> None:
        if not callable(prepare):
            raise TypeError("prepare must be callable")
        self._spec = spec
        self.instance_label = str(spec.title)
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
        self._request = (
            spec.build_request(self._values)
            if frozen_request is _BUILD_REQUEST
            else frozen_request
        )
        # Freeze the output vocabulary beside the request.  Camera output
        # cardinality is request state (frames_per_cycle), not a late guess from
        # whichever Dataset revision happens to reach the GUI first.
        self._output_declarations = tuple(spec.outputs_for(self._values))
        self._handle = None
        self._start_future: Future | None = None
        self._start_pending = False
        self._cancelled_before_start = False
        self._stop_event = threading.Event()
        self._snapshot = None
        # A FINAL value belongs to exactly one started generation.  It is
        # deliberately not inferred from the terminal snapshot: only
        # RunHandle.result() carries the value committed by the Run.
        self._final_result = _UNRESOLVED_FINAL
        self._final_projector = None
        self._final_projection_submitted = False
        self._projected_final_signals = None
        self._final_projection_error: str | None = None
        self._error: str | None = None
        self._start_exception: BaseException | None = None
        self._stop_requested = False
        self._stop_reason = "Console user requested stop"
        self._starter = None

    # ----------------------------------------------------------------- facts
    @property
    def spec(self):
        return self._spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def display_label(self) -> str:
        """The catalog title shown wherever the console names this producer."""

        return self._spec.title

    @property
    def layer(self) -> str:
        """Measurement / processor / task is a Definition fact, not runtime state."""

        return self._spec.kind

    @property
    def prefix(self) -> str:
        """Qualified signals no longer use mutable Hub-era prefixes."""

        return ""

    def signal_key(self, output_name: str) -> str:
        """Exact panel-binding key for one definition-owned output."""

        return console_signal_key(self.instance_label, output_name)

    def published_signals(self) -> tuple[str, ...]:
        """Return the exact producer-instance outputs published by this node.

        The short names remain owned by the catalog.  Qualifying them with the
        saved row title prevents two valid producers of ``frame`` or ``counts``
        from overwriting one another in the board data plane.
        """

        return tuple(
            self.signal_key(item.name) for item in self._output_declarations
        )

    @property
    def output_declarations(self) -> tuple:
        """The exact output vocabulary frozen with this node's request."""

        return self._output_declarations

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
    def start_exception(self) -> BaseException | None:
        """Structured exception from the most recent start submission.

        Normal presentation uses :attr:`last_error`.  The TaskConsole shell also
        needs an exact ``RunStartRejected`` payload to retire a conflicting Logic
        row without parsing text or weakening runtime admission.
        """

        return self._start_exception

    @property
    def final_result(self):
        """The successful Run result, or ``None`` until/when none exists.

        The property never waits.  :meth:`poll` admits the result only after
        the matching handle reports ``SUCCEEDED`` and its owner thread has
        finished.  Starting another generation clears it before submission,
        so a panel can never keep offering an earlier artifact while a rerun
        is in flight.
        """

        return None if self._final_result is _UNRESOLVED_FINAL else self._final_result

    @property
    def final_result_resolved(self) -> bool:
        """Whether the successful Run result has been joined without waiting.

        ``None`` may itself be a legitimate successful result, so callers must
        not use :attr:`final_result` as a completion flag.  This property is the
        narrow distinction needed by the GUI owner: keep polling a terminal
        Run until its owner thread has exited, then detach it.
        """

        return self._final_result is not _UNRESOLVED_FINAL

    @property
    def final_projection_resolved(self) -> bool:
        """Whether the optional FINAL dataset projection has finished.

        A committed artifact remains a successful result even when its
        presentation projection fails.  The separate projection error is shown
        by the Workbench without rewriting Run terminal truth.
        """

        return (
            self._final_result is not _UNRESOLVED_FINAL
            and (
                self._final_projector is None
                or self._projected_final_signals is not None
            )
        )

    @property
    def projected_final_signals(self):
        if self._projected_final_signals is None:
            return None
        return dict(self._projected_final_signals)

    @property
    def final_projection_error(self) -> str | None:
        return self._final_projection_error

    @property
    def running(self) -> bool:
        # A start is already this node's accepted lifecycle generation before
        # the owner-thread completion installs its RunHandle.  Live view ports
        # may publish their first immutable revision in that narrow interval;
        # treating the producer as stopped makes a real same-node publication
        # impossible to bind reactively until an unrelated GUI poll occurs.
        if self._start_pending:
            return True
        snapshot = self._snapshot
        if self._handle is None:
            return False
        return snapshot is None or not snapshot.state.terminal

    @property
    def lifecycle_generation(self) -> int:
        """Owner generation that identifies the currently accepted start.

        This is deliberately distinct from a dataset stream generation.  It
        lets a downstream reactive node pin the source Run even while its first
        frame has arrived just before the corresponding RunHandle completion is
        drained on the Qt owner thread.
        """

        return self._owner.generation

    @property
    def worker_idle(self) -> bool:
        """Whether this node has no pending prepare/start/projection callback."""

        return self._owner.worker_idle

    @property
    def cancelled_before_start(self) -> bool:
        """Whether Stop won before a domain RunHandle was created."""

        return self._cancelled_before_start

    # ------------------------------------------------------------ lifecycle
    def bind_starter(self, start: Callable[[object], object]) -> None:
        """Fix how THIS node starts, so a caller with no such knowledge can start it.

        The composition root knows a camera monitor needs a live view and what
        that view is; the board's Start button knows only "start this row".
        Binding once at construction lets the board call :meth:`start` with no
        argument and still get the right shape.
        """

        if not callable(start):
            raise TypeError("starter must be callable")
        self._starter = start

    def bind_final_projector(
        self,
        projector: Callable[[object], object],
    ) -> None:
        """Bind the composition-owned FINAL artifact -> display projection.

        Projection may load a large artifact, so it always runs on this node's
        worker after the Run result is available; the Qt owner only admits the
        completed immutable snapshots.
        """

        if not callable(projector):
            raise TypeError("final projector must be callable")
        if self._final_projector is not None:
            raise RuntimeError("this node already has a final projector")
        self._final_projector = projector

    def start(self, start: Callable[[object], object] | None = None) -> None:
        """Prepare and start on the worker.  Returns immediately.

        ``start`` receives the prepared command and returns its RunHandle.  How a
        command starts is the command's own business -- a camera monitor only
        starts with its live view, while a finite capture just starts
        -- so this seam sequences the call rather than knowing the shapes.

        One submission does both prepare and start: a prepared command that is
        never started is a reservation the operator did not ask for, and
        splitting them lets a stale prepare outlive the request it was frozen
        from.
        """

        start = self._starter if start is None else start
        if not callable(start):
            raise TypeError(
                "this node has no starter: bind one (bind_starter) or pass it here"
            )
        if self.running or self._start_pending:
            return
        self._error = None
        self._start_exception = None
        self._stop_requested = False
        self._cancelled_before_start = False
        self._stop_event.clear()
        self._stop_reason = "Console user requested stop"
        generation = self._owner.begin_generation()
        self._handle = None
        self._snapshot = None
        self._final_result = _UNRESOLVED_FINAL
        self._final_projection_submitted = False
        self._projected_final_signals = None
        self._final_projection_error = None
        request = self._request
        prepare = self._prepare

        def start_owned():
            prepared = prepare(request)
            if self._stop_event.is_set():
                raise _StartSuppressed()
            return start(prepared)

        future = self._owner.submit(
            "start",
            start_owned,
            generation=generation,
        )
        self._start_future = future
        self._start_pending = True

    def cancel(self, reason: str = "Console user requested stop") -> None:
        """Ask the Run to stop.  Never waits for the worker to reap it."""

        handle = self._handle
        self._stop_requested = True
        self._stop_event.set()
        self._stop_reason = str(reason)
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
            if completion.kind == "project-final":
                error = completion.future.exception()
                if error is None:
                    projected = completion.future.result()
                    if not isinstance(projected, dict):
                        self._final_projection_error = (
                            "TypeError: final projector must return a dict"
                        )
                        self._projected_final_signals = {}
                    else:
                        self._projected_final_signals = dict(projected)
                else:
                    self._final_projection_error = (
                        f"{type(error).__name__}: {error}"
                    )
                    self._projected_final_signals = {}
                continue
            error = completion.future.exception()
            if error is not None:
                self._start_pending = False
                if isinstance(error, _StartSuppressed):
                    self._cancelled_before_start = True
                    self._handle = None
                    self._snapshot = None
                    self._final_result = _UNRESOLVED_FINAL
                    self._owner.mark_owner_reaped()
                    continue
                self._error = f"{type(error).__name__}: {error}"
                self._start_exception = error
                self._handle = None
                self._snapshot = None
                self._final_result = _UNRESOLVED_FINAL
                self._owner.mark_owner_reaped()
                continue
            if completion.kind == "start":
                self._start_pending = False
                self._handle = completion.future.result()
                self._owner.set_handle(self._handle)
                if self._stop_requested:
                    self._handle.cancel(self._stop_reason)
        handle = self._handle
        if handle is not None:
            self._snapshot = handle.snapshot()
            if self._snapshot.state.terminal:
                if (
                    self._snapshot.state.name == "SUCCEEDED"
                    and self._final_result is _UNRESOLVED_FINAL
                ):
                    try:
                        # A Run publishes terminal state just before its owner
                        # thread exits.  timeout=0 keeps the Qt polling path
                        # non-blocking; the next tick retries that narrow join.
                        self._final_result = handle.result(timeout=0.0)
                    except TimeoutError:
                        pass
                    else:
                        self._owner.mark_owner_reaped()
                        projector = self._final_projector
                        if (
                            projector is not None
                            and not self._final_projection_submitted
                        ):
                            result = self._final_result
                            self._final_projection_submitted = True
                            self._owner.submit(
                                "project-final",
                                lambda: projector(result),
                                generation=self._owner.generation,
                            )
                elif self._snapshot.state.name != "SUCCEEDED":
                    self._final_result = _UNRESOLVED_FINAL
                    self._owner.mark_owner_reaped()
        return self._snapshot

    def wait_until_terminal(self, *, reason: str) -> None:
        """Join this node's already-submitted Run without draining Qt completions.

        ``Future`` and ``RunHandle`` are the two thread-safe ownership edges.
        GUI-owned snapshots and final projection remain exclusively in
        :meth:`poll`.  The bound Port/Run owns its finite I/O deadlines; this
        join does not invent a second user-tunable timeout.
        """

        future = self._start_future
        if future is None:
            return
        try:
            handle = future.result()
        except _StartSuppressed:
            return
        handle.cancel(str(reason))
        handle.wait()

    def shutdown(self) -> None:
        self.cancel("Console is closing")
        self._owner.shutdown()
