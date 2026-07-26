"""Raw live-dataset attachment owned by one Workbench consumer.

The slot carries acquisition revisions across the worker/UI ownership boundary.
It deliberately has no selector, ROI, reduction, Fit, or render policy: those
are Figure-owned branches over an accepted immutable front.
"""

from __future__ import annotations

import threading
from typing import Callable, Mapping

from zlc_frontend.figure import DatasetId
from zlc_neutral_atom.dataset_output import (
    LiveDatasetOutput,
    LiveDatasetOutputOwner,
    LiveDatasetSnapshotSource,
)
from zlc_neutral_atom.runtime.dataset import MonitorDatasetSnapshot
from zlc_neutral_atom.runtime.preview import LiveDatasetViewSpec
from zlc_storage import canonical_text


class LiveDatasetSlot:
    """One materializer lifetime plus coalesced revision notifications."""

    def __init__(
        self,
        spec: LiveDatasetViewSpec,
        *,
        dataset_id: DatasetId,
        retain_on_terminal: bool = True,
        output_owner: LiveDatasetOutputOwner | None = None,
    ) -> None:
        if not isinstance(spec, LiveDatasetViewSpec):
            raise TypeError("spec must implement LiveDatasetViewSpec")
        if not isinstance(dataset_id, DatasetId):
            raise TypeError("dataset_id must be DatasetId")
        if not isinstance(retain_on_terminal, bool):
            raise TypeError("retain_on_terminal must be bool")
        if output_owner is not None and not callable(
            getattr(output_owner, "live_dataset_outputs", None)
        ):
            raise TypeError(
                "output_owner must implement the neutral live output contract"
            )
        self.spec = spec
        self.dataset_id = dataset_id
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


__all__ = ["LiveDatasetSlot"]
