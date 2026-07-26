"""Headless hosted Run lifecycle, off the caller thread.

A hosted run is a frozen typed request plus a Run.  The caller freezes the
request; this module owns what happens next -- prepare, start, cancel, and the
latest observable snapshot.

The caller never blocks on domain preparation or start. Every round trip is
submitted to :class:`RunOwnerMailbox`; an owner loop drains completions.

Layering: this module does NOT import an application facade or UI toolkit.  It
takes a ``prepare`` callable from the composition root, so this lifecycle stays
free of domain and presentation authority.
"""

from __future__ import annotations

from concurrent.futures import Future
import threading
from typing import Callable

from zlc_neutral_atom.artifact_output import ArtifactOutputDeclaration
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from .owner_mailbox import RunOwnerMailbox
from zlc_neutral_atom.runtime.signal_source import SignalEventSource

__all__ = ["HostedRun"]


_UNRESOLVED_FINAL = object()


class _StartSuppressed(Exception):
    """The user stopped this node before its starter received hardware authority."""


class HostedRun:
    """The lifecycle of one hosted run.

    ``prepare`` receives the node's frozen request and returns the domain's
    prepared command.  The composition root binds the command's concrete start
    operation once; this owner sequences it without interpreting devices,
    outputs, or presentation.
    """

    def __init__(
        self,
        *,
        definition_key: DefinitionKey,
        request: object,
        instance_id: str,
        dataset_output_declarations: tuple[DatasetOutputDeclaration, ...],
        artifact_output_declarations: tuple[ArtifactOutputDeclaration, ...] = (),
        prepare: Callable[[object], object],
        qualify_output: Callable[[str], str],
        request_owner_wake: Callable[[], None],
    ) -> None:
        if not isinstance(definition_key, DefinitionKey):
            raise TypeError("definition_key must be DefinitionKey")
        if not callable(prepare):
            raise TypeError("prepare must be callable")
        if not callable(qualify_output):
            raise TypeError("qualify_output must be callable")
        identity = str(instance_id).strip()
        if not identity:
            raise ValueError("hosted instance id must be non-empty")
        dataset_outputs = tuple(dataset_output_declarations)
        if any(
            not isinstance(value, DatasetOutputDeclaration)
            for value in dataset_outputs
        ):
            raise TypeError(
                "dataset_output_declarations must contain "
                "DatasetOutputDeclaration values"
            )
        artifact_outputs = tuple(artifact_output_declarations)
        if any(
            not isinstance(value, ArtifactOutputDeclaration)
            for value in artifact_outputs
        ):
            raise TypeError(
                "artifact_output_declarations must contain "
                "ArtifactOutputDeclaration values"
            )
        dataset_names = tuple(value.name for value in dataset_outputs)
        artifact_names = tuple(value.name for value in artifact_outputs)
        if len(set(dataset_names)) != len(dataset_names):
            raise ValueError("Dataset output names must be unique")
        if len(set(artifact_names)) != len(artifact_names):
            raise ValueError("Artifact output names must be unique")
        if set(dataset_names).intersection(artifact_names):
            raise ValueError("Dataset and Artifact output names must not overlap")
        self._definition_key = definition_key
        self.instance_id = identity
        self._prepare = prepare
        self._qualify_output = qualify_output
        # One worker thread per node: a node's prepare/start/cancel jobs must not
        # interleave with each other, and thread affinity is what lets a domain
        # port that is not thread-safe be driven from here at all.
        self._owner = RunOwnerMailbox(
            request_owner_wake,
            thread_name_prefix=f"hosted-{definition_key.stable_definition_id}",
            max_workers=1,
        )
        self._request = request
        self._dataset_output_declarations = dataset_outputs
        self._artifact_output_declarations = artifact_outputs
        self._prepared_command = None
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
        self._final_outputs_submitted = False
        self._materialized_final_outputs = None
        self._completion_summary: str | None = None
        self._final_output_error: str | None = None
        self._error: str | None = None
        self._start_exception: BaseException | None = None
        self._stop_requested = False
        self._stop_reason = "Host requested stop"
        self._starter = None

    # ----------------------------------------------------------------- facts
    @property
    def definition_key(self) -> DefinitionKey:
        return self._definition_key

    def signal_key(self, output_name: str) -> str:
        """Exact host route for one definition-owned output."""

        return self._qualify_output(output_name)

    def published_signals(self) -> tuple[str, ...]:
        """Return the exact producer-instance outputs published by this node.

        The short names remain owned by the domain applications.  The injected
        qualifier supplies their stable host routes, preventing two valid
        producers of ``frame`` or ``counts`` from overwriting one another.
        """

        return tuple(
            self.signal_key(item.name)
            for item in self._dataset_output_declarations
        )

    def published_artifacts(self) -> tuple[str, ...]:
        """Return exact keys for FINAL artifact outputs, never Dataset signals."""

        return tuple(
            self.signal_key(item.name)
            for item in self._artifact_output_declarations
        )

    @property
    def dataset_output_declarations(
        self,
    ) -> tuple[DatasetOutputDeclaration, ...]:
        return self._dataset_output_declarations

    @property
    def artifact_output_declarations(
        self,
    ) -> tuple[ArtifactOutputDeclaration, ...]:
        return self._artifact_output_declarations

    @property
    def request(self):
        """The frozen typed request -- the node's identity for this run."""

        return self._request

    def value_schema(self, output_name: str):
        """Delegate one running producer's declared live-event schema."""

        command = self._prepared_command
        if command is None:
            raise RuntimeError("producer has not finished prepare/start submission")
        resolve = getattr(command, "value_schema", None)
        if not callable(resolve):
            raise TypeError("this producer does not expose live signal events")
        return resolve(output_name)

    def open_signal_cursor(self, output_name: str):
        """Open a future-only cursor without exposing the prepared command."""

        command = self._prepared_command
        if command is None:
            raise RuntimeError("producer has not finished prepare/start submission")
        open_cursor = getattr(command, "open_signal_cursor", None)
        if not callable(open_cursor):
            raise TypeError("this producer does not expose live signal events")
        return open_cursor(output_name)

    def dataset_output_binding(self, output_name: str):
        """Return an optional domain-owned binding for one running Dataset output."""

        command = self._prepared_command
        if command is None:
            raise RuntimeError("producer has not finished prepare/start submission")
        resolve = getattr(command, "dataset_output_binding", None)
        return None if not callable(resolve) else resolve(output_name)

    def signal_event_source(self) -> SignalEventSource:
        """Return the prepared command's real source without proxy capabilities."""

        if not self.running:
            raise RuntimeError("producer signal source is not running")
        command = self._prepared_command
        if not isinstance(command, SignalEventSource):
            raise TypeError("this producer does not expose live signal events")
        return command

    @property
    def handle(self):
        return self._handle

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def start_exception(self) -> BaseException | None:
        """Structured exception from the most recent start submission.

        Normal host status uses :attr:`last_error`; typed resource admission
        is normalized separately by :attr:`resource_conflict`.
        """

        return self._start_exception

    @property
    def resource_conflict(self):
        """Return one typed admission conflict across direct/composite starts.

        ``RunStartRejected`` belongs to the synchronous controller start API.
        A composite handle instead carries the same ``ResourceBusy`` outcome
        on its immutable snapshot.  Normalizing both here keeps every host
        independent of exception timing and forbids RunId parsing from text.
        """

        from zlc_neutral_atom.runtime.resources import ResourceBusy
        from zlc_neutral_atom.runtime.run import RunStartRejected

        direct = self._start_exception
        if isinstance(direct, RunStartRejected):
            return direct.outcome
        snapshot = self._snapshot
        conflict = (
            None
            if snapshot is None
            else getattr(snapshot, "admission_rejection", None)
        )
        if conflict is not None and not isinstance(conflict, ResourceBusy):
            raise TypeError("RunSnapshot admission_rejection must be ResourceBusy")
        return conflict

    @property
    def final_result(self):
        """The successful Run result, or ``None`` until/when none exists.

        The property never waits.  :meth:`poll` admits the result only after
        the matching handle reports ``SUCCEEDED`` and its owner thread has
        finished.  Starting another generation clears it before submission,
        so a consumer can never keep offering an earlier artifact while a rerun
        is in flight.
        """

        return None if self._final_result is _UNRESOLVED_FINAL else self._final_result

    @property
    def prepared_command(self) -> object | None:
        """Return the command admitted for this generation, without ownership."""

        return self._prepared_command

    @property
    def final_result_resolved(self) -> bool:
        """Whether the successful Run result has been joined without waiting.

        ``None`` may itself be a legitimate successful result, so callers must
        not use :attr:`final_result` as a completion flag.  This property is the
        narrow distinction needed by the host owner: keep polling a terminal
        Run until its owner thread has exited, then detach it.
        """

        return self._final_result is not _UNRESOLVED_FINAL

    @property
    def final_outputs_resolved(self) -> bool:
        """Whether the optional FINAL output materialization has finished.

        A committed artifact remains a successful result even when its named
        output materialization fails. That boundary error does not rewrite Run
        terminal truth.
        """

        return (
            self._final_result is not _UNRESOLVED_FINAL
            and (
                self._prepared_command is None
                or not callable(
                    getattr(
                        self._prepared_command,
                        "final_dataset_outputs",
                        None,
                    )
                )
                or self._materialized_final_outputs is not None
            )
        )

    @property
    def materialized_final_outputs(self):
        if self._materialized_final_outputs is None:
            return None
        return dict(self._materialized_final_outputs)

    @property
    def final_output_error(self) -> str | None:
        return self._final_output_error

    @property
    def completion_summary(self) -> str | None:
        """Command-owned human result location, if this run persists one."""

        return self._completion_summary

    @property
    def running(self) -> bool:
        # A start is already this node's accepted lifecycle generation before
        # the owner-thread completion installs its RunHandle.  Live view ports
        # may publish their first immutable revision in that narrow interval;
        # treating the producer as stopped makes a real same-node publication
        # impossible to bind reactively until an unrelated host poll occurs.
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
        drained by the owner loop.
        """

        return self._owner.generation

    @property
    def worker_idle(self) -> bool:
        """Whether this node has no pending prepare/start/output callback."""

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
        self._stop_reason = "Host requested stop"
        generation = self._owner.begin_generation()
        self._handle = None
        self._prepared_command = None
        self._snapshot = None
        self._final_result = _UNRESOLVED_FINAL
        self._final_outputs_submitted = False
        self._materialized_final_outputs = None
        self._completion_summary = None
        self._final_output_error = None
        request = self._request
        prepare = self._prepare

        def start_owned():
            prepared = prepare(request)
            if self._stop_event.is_set():
                raise _StartSuppressed()
            return prepared, start(prepared)

        future = self._owner.submit(
            "start",
            start_owned,
            generation=generation,
        )
        self._start_future = future
        self._start_pending = True

    def cancel(self, reason: str = "Host requested stop") -> None:
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
        Consumers read state from the returned snapshot rather than from a
        second flag, so lifecycle decisions and diagnostics cannot disagree
        about the same run.
        """

        for completion in self._owner.drain_completions():
            if completion.generation != self._owner.generation:
                continue          # a job from a superseded generation; its result is stale
            if completion.kind == "materialize-final-outputs":
                error = completion.future.exception()
                if error is None:
                    projected = completion.future.result()
                    if not isinstance(projected, dict):
                        self._final_output_error = (
                            "TypeError: final output owner must return a dict"
                        )
                        self._materialized_final_outputs = {}
                    else:
                        actual = set(map(str, projected))
                        if not actual:
                            self._final_output_error = (
                                "ValueError: final outputs factory returned no outputs"
                            )
                            self._materialized_final_outputs = {}
                        else:
                            self._materialized_final_outputs = dict(projected)
                else:
                    self._final_output_error = (
                        f"{type(error).__name__}: {error}"
                    )
                    self._materialized_final_outputs = {}
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
                prepared, self._handle = completion.future.result()
                self._prepared_command = prepared
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
                        # thread exits.  timeout=0 keeps observation
                        # non-blocking; the next poll retries that narrow join.
                        self._final_result = handle.result(timeout=0.0)
                    except TimeoutError:
                        pass
                    else:
                        self._owner.mark_owner_reaped()
                        command = self._prepared_command
                        summary_factory = getattr(
                            command,
                            "completion_summary",
                            None,
                        )
                        if callable(summary_factory):
                            try:
                                summary = summary_factory(self._final_result)
                                if not isinstance(summary, str) or not summary.strip():
                                    raise TypeError(
                                        "completion_summary() must return non-empty str"
                                    )
                                self._completion_summary = summary.strip()
                            except BaseException as error:
                                self._completion_summary = (
                                    "result committed; location unavailable: "
                                    f"{type(error).__name__}: {error}"
                                )
                        outputs_factory = getattr(
                            command,
                            "final_dataset_outputs",
                            None,
                        )
                        if (
                            callable(outputs_factory)
                            and not self._final_outputs_submitted
                        ):
                            result = self._final_result
                            def materialize_final_outputs():
                                return outputs_factory(result)

                            self._final_outputs_submitted = True
                            self._owner.submit(
                                "materialize-final-outputs",
                                materialize_final_outputs,
                                generation=self._owner.generation,
                            )
                elif self._snapshot.state.name != "SUCCEEDED":
                    self._final_result = _UNRESOLVED_FINAL
                    self._owner.mark_owner_reaped()
        return self._snapshot

    def shutdown(self) -> None:
        self.cancel("Host is closing")
        self._owner.shutdown()
