"""One Workbench-owned read-only attachment to an exact dataset builder."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

from zlc_data import DatasetRevision
from zlc_neutral_atom.runtime.dataset import (
    DatasetPreviewDelta,
    DatasetPreviewSnapshot,
    ExactDatasetPreviewReader,
)
from zlc_neutral_atom.runtime.preview import ExactDatasetPreviewSpec
from zlc_storage import canonical_text


class ExactDatasetLiveSlot:
    """Read-only exact preview lifetime shared by all Workbench projections."""

    def __init__(self, spec: ExactDatasetPreviewSpec) -> None:
        if not isinstance(spec, ExactDatasetPreviewSpec):
            raise TypeError("spec must be ExactDatasetPreviewSpec")
        self._spec = spec
        self._condition = threading.Condition(threading.Lock())
        self._reader: ExactDatasetPreviewReader | None = None
        self._run_id: str | None = None
        self._causation_domain_id: str | None = None
        self._listener: Callable[[], None] | None = None
        self._pending_change = False
        self._failure: str | None = None
        self._terminal = False
        self._closed = False

    @property
    def spec(self) -> ExactDatasetPreviewSpec:
        return self._spec

    @property
    def terminal(self) -> bool:
        with self._condition:
            return self._terminal

    @property
    def failure(self) -> str | None:
        with self._condition:
            return self._failure

    def set_change_listener(self, listener: Callable[[], None]) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        replay = False
        with self._condition:
            if self._listener is not None:
                raise RuntimeError("exact live slot already has a listener")
            if self._closed:
                raise RuntimeError("exact live slot is closed")
            self._listener = listener
            replay, self._pending_change = self._pending_change, False
        if replay:
            listener()

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
            raise ValueError("exact reader schema differs from preview spec")
        run_id = canonical_text(run_id, "run_id")
        causation_domain_id = canonical_text(
            causation_domain_id,
            "causation_domain_id",
        )
        if reader.stream_generation.value != causation_domain_id:
            raise ValueError("exact reader generation differs from causation domain")
        if reader.terminal:
            raise RuntimeError("exact reader is already terminal")
        listener = None
        with self._condition:
            if self._closed or self._terminal:
                raise RuntimeError("exact live slot is terminal")
            if self._reader is not None:
                raise RuntimeError("exact live slot is already bound")
            self._reader = reader
            self._run_id = run_id
            self._causation_domain_id = causation_domain_id
            self._condition.notify_all()
            listener = self._notify_locked()
        if listener is not None:
            listener()

    def wait_and_freeze(
        self,
        after: DatasetRevision,
        *,
        timeout: float | None,
    ) -> tuple[str, str, DatasetPreviewSnapshot] | None:
        identity = self._wait_for_new_revision(after, timeout)
        if identity is None:
            return None
        reader, run_id, causation = identity
        return run_id, causation, reader.freeze_current()

    def wait_and_freeze_delta(
        self,
        after: DatasetRevision,
        *,
        timeout: float | None,
    ) -> tuple[str, str, DatasetPreviewDelta] | None:
        """Freeze only newly committed cells after the caller's cursor."""

        identity = self._wait_for_new_revision(after, timeout)
        if identity is None:
            return None
        reader, run_id, causation = identity
        return run_id, causation, reader.freeze_delta(after)

    def fail(self, message: str) -> None:
        message = canonical_text(message, "message")
        listener = None
        with self._condition:
            if self._closed:
                return
            if self._failure is None:
                self._failure = message
            self._reader = None
            self._terminal = True
            self._condition.notify_all()
            listener = self._notify_locked()
        if listener is not None:
            listener()

    def source_terminal(self) -> None:
        with self._condition:
            if self._closed or self._terminal:
                return
            reader = self._reader
        try:
            if reader is None:
                raise RuntimeError("exact preview reached terminal before reader bind")
            if not reader.terminal:
                raise RuntimeError("exact preview source is not terminal")
            if reader.failed:
                raise RuntimeError("exact preview source aborted")
            if not reader.coverage.complete:
                raise RuntimeError("exact preview source terminal coverage is incomplete")
        except BaseException as error:
            self.fail(f"{type(error).__name__}: {error}")
            return
        listener = None
        with self._condition:
            if self._closed or self._terminal:
                return
            # Retain the read-only reader.  A cumulative consumer may freeze
            # the terminal dataset once; a delta consumer copies only unseen
            # cells.  Merely validating terminal never copies frame values.
            self._terminal = True
            self._condition.notify_all()
            listener = self._notify_locked()
        if listener is not None:
            listener()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._terminal = True
            self._reader = None
            self._listener = None
            self._condition.notify_all()

    def _wait_for_new_revision(
        self,
        after: DatasetRevision,
        timeout: float | None,
    ) -> tuple[ExactDatasetPreviewReader, str, str] | None:
        if not isinstance(after, DatasetRevision):
            raise TypeError("after must be DatasetRevision")
        if timeout is None:
            deadline = None
        else:
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(float(timeout))
                or float(timeout) < 0
            ):
                raise ValueError("timeout must be finite and non-negative or None")
            deadline = time.monotonic() + float(timeout)
        while True:
            with self._condition:
                if self._failure is not None:
                    raise RuntimeError(self._failure)
                if self._closed:
                    return None
                reader = self._reader
                run_id = self._run_id
                causation = self._causation_domain_id
                terminal = self._terminal
            if reader is None:
                return None
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            revision = reader.wait_for_change(after, remaining)
            if revision is not None:
                assert run_id is not None and causation is not None
                return reader, run_id, causation
            # Builder seal is not yet the Workbench terminal: cleanup and
            # post-safety still decide whether this preview is successful.
            # Waiting on the slot condition avoids a hot loop while the sealed
            # reader itself returns immediately from wait_for_change().
            if terminal or (
                deadline is not None and time.monotonic() >= deadline
            ):
                return None
            with self._condition:
                predicate = lambda: (
                    self._failure is not None
                    or self._terminal
                    or self._closed
                    or self._reader is not reader
                )
                if deadline is None:
                    self._condition.wait_for(predicate)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait_for(predicate, remaining)

    def _notify_locked(self) -> Callable[[], None] | None:
        listener = self._listener
        if listener is None:
            self._pending_change = True
        return listener


__all__ = ["ExactDatasetLiveSlot"]
