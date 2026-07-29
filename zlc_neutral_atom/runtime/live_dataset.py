"""Raw live-dataset attachment owned by one host consumer.

The slot carries acquisition revisions across the producer/consumer ownership boundary.
It deliberately has no projection, analysis, or render policy: those are
downstream branches over an accepted immutable front.
"""

from __future__ import annotations

import threading
from typing import Callable, Mapping

from zlc_data import DatasetRevision
from zlc_neutral_atom.dataset_output import (
    LiveDatasetOutput,
    LiveDatasetOutputOwner,
    LiveDatasetSnapshotSource,
)
from zlc_neutral_atom.runtime.dataset import (
    ExactDatasetPreviewReader,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewSpec,
    LiveDatasetViewSpec,
)
from zlc_storage import canonical_text

from ._failure import safe_error_summary


class LiveDatasetPort:
    """One materializer lifetime plus coalesced revision notifications."""

    def __init__(
        self,
        spec: LiveDatasetViewSpec,
        *,
        retain_on_terminal: bool = True,
        output_owner: LiveDatasetOutputOwner | None = None,
    ) -> None:
        if not isinstance(spec, LiveDatasetViewSpec):
            raise TypeError("spec must implement LiveDatasetViewSpec")
        if not isinstance(retain_on_terminal, bool):
            raise TypeError("retain_on_terminal must be bool")
        if output_owner is not None and not callable(
            getattr(output_owner, "live_dataset_outputs", None)
        ):
            raise TypeError(
                "output_owner must implement the neutral live output contract"
            )
        self.spec = spec
        self._retain_on_terminal = retain_on_terminal
        self._output_owner = output_owner
        self._lock = threading.Lock()
        self._dataset: LiveDatasetSnapshotSource | None = None
        self._run_id: str | None = None
        self._causation_domain_id: str | None = None
        self._listener: Callable[[], None] | None = None
        self._listener_claimed = False
        self._pending_change = False
        self._failure: str | None = None
        self._notification_failure: str | None = None
        self._terminal = False
        self._withdrawn = False
        self._closed = False

    @property
    def failure(self) -> str | None:
        with self._lock:
            return self._failure

    @property
    def notification_failure(self) -> str | None:
        with self._lock:
            return self._notification_failure

    @property
    def terminal(self) -> bool:
        with self._lock:
            return self._terminal

    @property
    def dataset_bound(self) -> bool:
        with self._lock:
            return not self._closed and self._dataset is not None

    @property
    def withdrawn(self) -> bool:
        with self._lock:
            return self._withdrawn

    def set_change_listener(self, listener: Callable[[], None]) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        replay = False
        with self._lock:
            if self._listener_claimed:
                raise RuntimeError("live slot already has a change listener")
            if self._closed:
                raise RuntimeError("live slot is closed")
            self._listener_claimed = True
            self._listener = listener
            replay, self._pending_change = self._pending_change, False
        if replay:
            listener()

    def bind(
        self,
        dataset: LiveDatasetSnapshotSource,
        *,
        run_id: str,
        causation_domain_id: str,
    ) -> None:
        if not isinstance(dataset, LiveDatasetSnapshotSource):
            raise TypeError("dataset must implement LiveDatasetSnapshotSource")
        run_id = canonical_text(run_id, "run_id")
        causation_domain_id = canonical_text(
            causation_domain_id,
            "causation_domain_id",
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("live slot is closed")
            if self._terminal:
                raise RuntimeError("live slot is terminal")
            if self._dataset is not None:
                raise RuntimeError("live slot already owns a materializer")
            self._dataset = dataset
            self._run_id = run_id
            self._causation_domain_id = causation_domain_id

    def updated(self) -> None:
        with self._lock:
            if self._closed or self._dataset is None:
                raise RuntimeError("live slot has no active materializer")
            listener = self._listener
            if listener is None:
                self._pending_change = True
                return
        listener()

    def notification_failed(self, message: str) -> None:
        message = canonical_text(message, "live notification failure")
        with self._lock:
            if self._closed or self._dataset is None:
                return
            if self._failure is not None or self._notification_failure is not None:
                return
            self._notification_failure = message
            listener = self._listener
            if listener is None:
                self._pending_change = True
        if listener is not None:
            try:
                listener()
            except BaseException:
                pass

    def freeze_current(self) -> tuple[str, str, MonitorDatasetSnapshot]:
        with self._lock:
            if self._closed or self._dataset is None:
                raise RuntimeError("live slot has no active materializer")
            dataset = self._dataset
            run_id = self._run_id
            causation = self._causation_domain_id
        snapshot = dataset.freeze_current()
        with self._lock:
            if self._closed or self._dataset is not dataset:
                raise RuntimeError("live slot lifetime ended while freezing a snapshot")
        assert run_id is not None and causation is not None
        return run_id, causation, snapshot

    def freeze_live_outputs(
        self,
    ) -> tuple[str, str, Mapping[str, LiveDatasetOutput]]:
        """Delegate naming/materialization to the frozen application owner."""

        owner = self._output_owner
        if owner is None:
            raise RuntimeError("live slot has no application output owner")
        run_id, causation, snapshot = self.freeze_current()
        return run_id, causation, owner.live_dataset_outputs(snapshot)

    def fail(self, message: str) -> None:
        message = canonical_text(message, "preview failure")
        dataset, listener = self._detach(message)
        try:
            if dataset is not None:
                dataset.close()
        finally:
            if listener is not None:
                try:
                    listener()
                except BaseException:
                    pass

    def source_terminal(self) -> None:
        if self._retain_on_terminal:
            with self._lock:
                self._terminal = True
            return
        dataset, listener = self._detach(None, withdrawn=True)
        try:
            if dataset is not None:
                dataset.close()
        finally:
            if listener is not None:
                try:
                    listener()
                except BaseException:
                    pass

    def close(self) -> None:
        dataset, _listener = self._detach(None, closed=True)
        if dataset is not None:
            dataset.close()

    def _detach(
        self,
        failure: str | None,
        *,
        closed: bool = False,
        withdrawn: bool = False,
    ) -> tuple[LiveDatasetSnapshotSource | None, Callable | None]:
        with self._lock:
            if closed and self._closed:
                return None, None
            dataset, self._dataset = self._dataset, None
            if failure is not None:
                self._failure = failure
            self._terminal = True
            self._withdrawn = self._withdrawn or withdrawn
            self._closed = self._closed or closed
            listener = self._listener
            if closed:
                self._listener = None
            elif listener is None and not self._listener_claimed:
                self._pending_change = True
            return dataset, listener


class _ExactDeltaLivePort:
    """One internal exact-delta worker shared by finite Dataset products.

    The exact ``DatasetBuilder`` remains the artifact authority.  This port owns
    only a read-only revision cursor and one domain projector.  SignalDataPlane
    is the sole wake-coalescing owner.  The port never waits on the builder's
    lock condition: producers announce a successful commit through
    :meth:`updated`, so :meth:`close` can always wake and join its own worker.
    This port coalesces only source notices; SignalDataPlane remains the sole
    publication-frontier and consumer-wake authority.
    """

    def __init__(self, spec: ExactDatasetPreviewSpec, projection: object) -> None:
        if not isinstance(spec, ExactDatasetPreviewSpec):
            raise TypeError("spec must be ExactDatasetPreviewSpec")
        if not callable(getattr(projection, "consume", None)):
            raise TypeError("exact live projection must expose consume(delta)")
        if not callable(getattr(projection, "freeze_live_outputs", None)):
            raise TypeError(
                "exact live projection must expose freeze_live_outputs()"
            )
        self.spec = spec
        self._projection = projection
        self._projection_lock = threading.Lock()
        self._condition = threading.Condition(threading.Lock())
        self._reader: ExactDatasetPreviewReader | None = None
        self._run_id: str | None = None
        self._causation_domain_id: str | None = None
        self._listener: Callable[[], None] | None = None
        self._pending_source_change = False
        self._source_terminal = False
        self._failure: str | None = None
        self._notification_failure: str | None = None
        self._terminal = False
        self._closed = False
        self._after = DatasetRevision(0)
        self._worker = threading.Thread(
            target=self._watch,
            name="zlc-exact-delta-live",
            daemon=True,
        )
        self._worker.start()

    @property
    def terminal(self) -> bool:
        with self._condition:
            return self._terminal

    @property
    def notification_failure(self) -> str | None:
        with self._condition:
            return self._notification_failure

    def bind(
        self,
        reader: ExactDatasetPreviewReader,
        *,
        run_id: str,
        causation_domain_id: str,
    ) -> None:
        if not isinstance(reader, ExactDatasetPreviewReader):
            raise TypeError("reader must be ExactDatasetPreviewReader")
        if reader.schema.fingerprint != self.spec.source_schema_fingerprint:
            raise ValueError("exact live reader has another source schema")
        run_id = canonical_text(run_id, "run_id")
        causation_domain_id = canonical_text(
            causation_domain_id,
            "causation_domain_id",
        )
        if reader.stream_generation.value != causation_domain_id:
            raise ValueError("exact live generation differs from causation domain")
        if reader.terminal:
            raise RuntimeError("exact live reader is already terminal")
        with self._condition:
            if self._closed or self._terminal:
                raise RuntimeError("exact live port is terminal")
            if self._reader is not None:
                raise RuntimeError("exact live port is already bound")
            self._reader = reader
            self._run_id = run_id
            self._causation_domain_id = causation_domain_id
            self._condition.notify_all()

    def updated(self) -> None:
        """Wake the delta worker after one authoritative builder commit."""

        with self._condition:
            if self._closed or self._terminal or self._reader is None:
                raise RuntimeError("exact live port has no active reader")
            self._pending_source_change = True
            self._condition.notify_all()

    def set_change_listener(self, listener: Callable[[], None]) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        with self._condition:
            if self._closed:
                raise RuntimeError("exact live port is closed")
            if self._listener is not None:
                raise RuntimeError("exact live port already has a change listener")
            self._listener = listener

    def freeze_live_outputs(
        self,
    ) -> tuple[str, str, Mapping[str, LiveDatasetOutput]]:
        with self._condition:
            if self._closed:
                raise RuntimeError("exact live port is closed")
            if self._failure is not None:
                raise RuntimeError(self._failure)
            reader = self._reader
            run_id = self._run_id
            causation = self._causation_domain_id
            if reader is None or run_id is None or causation is None:
                raise RuntimeError("exact live port is not bound")
            if self._after.value < 1:
                raise RuntimeError("exact live port has no projected revision")
        with self._projection_lock:
            outputs = self._projection.freeze_live_outputs()
        with self._condition:
            if self._closed or self._reader is not reader:
                raise RuntimeError(
                    "exact live lifetime ended while freezing outputs"
                )
            if self._failure is not None:
                raise RuntimeError(self._failure)
        return run_id, causation, outputs

    def fail(self, message: str) -> None:
        message = canonical_text(message, "preview failure")
        with self._condition:
            if self._closed or self._terminal:
                return
            self._failure = message
            self._terminal = True
            self._condition.notify_all()
            listener = self._listener
        self._call_listener(listener)

    def source_terminal(self) -> None:
        with self._condition:
            if self._closed or self._terminal or self._source_terminal:
                return
            if self._reader is None:
                failure = "exact live source ended before reader bind"
            else:
                failure = None
                self._source_terminal = True
                self._pending_source_change = True
                self._condition.notify_all()
        if failure is not None:
            self.fail(failure)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._terminal = True
            self._reader = None
            self._listener = None
            self._condition.notify_all()
        if threading.current_thread() is not self._worker:
            self._worker.join()

    def _watch(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            self._closed
                            or self._failure is not None
                            or (
                                self._reader is not None
                                and (
                                    self._pending_source_change
                                    or self._source_terminal
                                )
                            )
                        )
                    )
                    if self._closed or self._failure is not None:
                        return
                    reader = self._reader
                    self._pending_source_change = False
                assert reader is not None
                while True:
                    with self._condition:
                        if self._closed or self._failure is not None:
                            return
                        if self._reader is not reader:
                            raise RuntimeError("exact live reader changed")
                        after = self._after
                    delta = reader.freeze_delta(after)
                    if not delta.cells:
                        break
                    # The reader freezes exactly one physical cell.  A close
                    # can therefore revoke a backlog between cells instead of
                    # joining one unbounded image-copy batch.
                    with self._condition:
                        if self._closed or self._failure is not None:
                            return
                        if self._reader is not reader:
                            raise RuntimeError("exact live reader changed")
                    with self._projection_lock:
                        publishable = self._projection.consume(delta)
                        if type(publishable) is not bool:
                            raise TypeError(
                                "exact live projection consume() must return bool"
                            )
                        with self._condition:
                            if self._closed or self._failure is not None:
                                return
                            if self._reader is not reader:
                                raise RuntimeError("exact live reader changed")
                            self._after = delta.ref.revision
                            listener = self._listener if publishable else None
                    self._call_listener(listener)
                with self._condition:
                    if self._closed or self._failure is not None:
                        return
                    if self._reader is not reader:
                        raise RuntimeError("exact live reader changed")
                    if self._pending_source_change:
                        continue
                    finish = self._source_terminal
                    current = self._after
                if not finish:
                    continue
                if not reader.terminal:
                    raise RuntimeError("exact live source is not terminal")
                if reader.failed:
                    raise RuntimeError("exact live source aborted")
                coverage = reader.coverage
                if not coverage.complete:
                    raise RuntimeError("exact live terminal coverage is incomplete")
                if current.value != coverage.written_cells:
                    raise RuntimeError("exact live worker did not drain terminal cells")
                with self._condition:
                    if self._closed or self._failure is not None:
                        return
                    self._terminal = True
                    self._condition.notify_all()
                return
        except BaseException as error:
            self.fail(safe_error_summary(error))

    def _call_listener(self, listener: Callable[[], None] | None) -> None:
        if listener is None:
            return
        try:
            listener()
        except BaseException as error:
            with self._condition:
                if self._notification_failure is None:
                    self._notification_failure = safe_error_summary(error)


__all__ = ["LiveDatasetPort"]
