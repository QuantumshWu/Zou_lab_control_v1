"""Run-scoped provisional output for one prepared MOT-field task.

The object in this module is both the exact acquisition's typed preview port and the
application-owned live-output source consumed by a frontend.  It never owns an
artifact or a GUI object: the exact DatasetBuilder remains authoritative while
the MOT projection publishes immutable scalar-grid revisions.
"""

from __future__ import annotations

import threading
from typing import Callable, Mapping

from zlc_data import DatasetRevision, DatasetSchema
from zlc_neutral_atom.dataset_output import LiveDatasetOutput
from .mot_field import MotFieldRequest
from .mot_field_live import MotFieldLiveProjection
from zlc_neutral_atom.runtime.dataset import (
    DatasetPreviewSnapshot,
    ExactDatasetPreviewReader,
)
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
)
from zlc_storage import canonical_text


class MotFieldTaskLiveOutput:
    """Project exact camera deltas into the task's one named live grid.

    ``PreparedMotFieldAcquisition`` sees this object only as an
    :class:`ExactDatasetPreviewPort`.  Workbench sees only
    ``set_change_listener``/``freeze_live_outputs``/``close``.  Neither side
    assembles or understands the projection pipeline.
    """

    def __init__(
        self,
        request: MotFieldRequest,
        source_schema: DatasetSchema,
    ) -> None:
        self._projection = MotFieldLiveProjection(
            request,
            source_schema,
        )
        self._spec = ExactDatasetPreviewSpec(source_schema.fingerprint)
        self._condition = threading.Condition(threading.Lock())
        self._reader: ExactDatasetPreviewReader | None = None
        self._run_id: str | None = None
        self._causation_domain_id: str | None = None
        self._listener: Callable[[], None] | None = None
        self._pending_change = False
        self._current: DatasetPreviewSnapshot | None = None
        self._failure: str | None = None
        self._notification_failure: str | None = None
        self._source_terminal = False
        self._terminal = False
        self._closed = False
        self._worker = threading.Thread(
            target=self._watch,
            name="zlc-mot-field-live-grid",
            daemon=True,
        )
        self._worker.start()

    @property
    def spec(self) -> ExactDatasetPreviewSpec:
        return self._spec

    @property
    def preview_port(self) -> ExactDatasetPreviewPort:
        """Return the same object narrowed to the scan preview protocol."""

        return self

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
        """Accept the exact builder's read-only preview exactly once."""

        if not isinstance(reader, ExactDatasetPreviewReader):
            raise TypeError("reader must be ExactDatasetPreviewReader")
        if reader.schema.fingerprint != self._spec.source_schema_fingerprint:
            raise ValueError("MOT preview reader has another source schema")
        run_id = canonical_text(run_id, "run_id")
        causation_domain_id = canonical_text(
            causation_domain_id,
            "causation_domain_id",
        )
        if reader.stream_generation.value != causation_domain_id:
            raise ValueError("MOT preview generation differs from causation domain")
        if reader.terminal:
            raise RuntimeError("MOT preview reader is already terminal")
        with self._condition:
            if self._closed or self._terminal:
                raise RuntimeError("MOT live output is terminal")
            if self._reader is not None:
                raise RuntimeError("MOT live output is already bound")
            self._reader = reader
            self._run_id = run_id
            self._causation_domain_id = causation_domain_id
            self._condition.notify_all()

    def fail(self, message: str) -> None:
        """Fail the provisional branch without changing the exact Run result."""

        message = canonical_text(message, "preview failure")
        listener = None
        with self._condition:
            if self._closed or self._terminal:
                return
            self._failure = message
            self._source_terminal = True
            self._terminal = True
            self._condition.notify_all()
            listener = self._notify_locked()
        self._call_listener(listener)

    def source_terminal(self) -> None:
        """Admit terminal only after the exact source proves full coverage."""

        with self._condition:
            if self._closed or self._terminal or self._source_terminal:
                return
            reader = self._reader
        try:
            if reader is None:
                raise RuntimeError("MOT preview ended before reader bind")
            if not reader.terminal:
                raise RuntimeError("MOT preview source is not terminal")
            if reader.failed:
                raise RuntimeError("MOT preview source aborted")
            if not reader.coverage.complete:
                raise RuntimeError("MOT preview terminal coverage is incomplete")
        except BaseException as error:
            self.fail(f"{type(error).__name__}: {error}")
            return
        with self._condition:
            if self._closed or self._terminal:
                return
            self._source_terminal = True
            self._condition.notify_all()

    def set_change_listener(self, listener: Callable[[], None]) -> None:
        """Install the frontend wake hook without transferring data ownership."""

        if not callable(listener):
            raise TypeError("listener must be callable")
        replay = False
        with self._condition:
            if self._listener is not None:
                raise RuntimeError("MOT live output already has a listener")
            if self._closed:
                raise RuntimeError("MOT live output is closed")
            self._listener = listener
            replay, self._pending_change = self._pending_change, False
        if replay:
            self._call_listener(listener)

    def freeze_current(self) -> tuple[str, str, DatasetPreviewSnapshot]:
        with self._condition:
            if self._closed:
                raise RuntimeError("MOT live output is closed")
            if self._failure is not None:
                raise RuntimeError(self._failure)
            if (
                self._run_id is None
                or self._causation_domain_id is None
                or self._current is None
            ):
                raise RuntimeError("MOT live output has no projected revision")
            return self._run_id, self._causation_domain_id, self._current

    def freeze_live_outputs(
        self,
    ) -> tuple[str, str, Mapping[str, LiveDatasetOutput]]:
        run_id, causation, frozen = self.freeze_current()
        return run_id, causation, self._projection.live_dataset_outputs(frozen)

    def close(self) -> None:
        """Withdraw the provisional route immediately and idempotently."""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._terminal = True
            self._reader = None
            self._listener = None
            self._condition.notify_all()

    def _watch(self) -> None:
        after = DatasetRevision(0)
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            self._reader is not None
                            or self._closed
                            or self._failure is not None
                        )
                    )
                    if self._closed or self._failure is not None:
                        return
                    reader = self._reader
                assert reader is not None
                revision = reader.wait_for_change(after, None)
                if revision is not None:
                    delta = reader.freeze_delta(after)
                    after = delta.ref.revision
                    if delta.cells:
                        self._consume_front(self._projection.consume(delta))
                    continue
                with self._condition:
                    if self._closed or self._failure is not None:
                        return
                    if self._source_terminal:
                        break
                    self._condition.wait_for(
                        lambda: (
                            self._closed
                            or self._failure is not None
                            or self._source_terminal
                            or self._reader is not reader
                        )
                    )
            self._finish()
        except BaseException as error:
            self.fail(f"{type(error).__name__}: {error}")

    def _consume_front(self, front: DatasetPreviewSnapshot) -> None:
        listener = None
        with self._condition:
            if self._closed or self._terminal:
                return
            self._current = front
            listener = self._notify_locked()
        self._call_listener(listener)

    def _finish(self) -> None:
        listener = None
        with self._condition:
            if self._closed or self._terminal:
                return
            self._terminal = True
            listener = self._notify_locked()
            self._condition.notify_all()
        self._call_listener(listener)

    def _notify_locked(self) -> Callable[[], None] | None:
        if self._listener is None:
            self._pending_change = True
        return self._listener

    def _call_listener(self, listener: Callable[[], None] | None) -> None:
        if listener is None:
            return
        try:
            listener()
        except BaseException as error:
            with self._condition:
                if self._notification_failure is None:
                    self._notification_failure = (
                        f"{type(error).__name__}: {error}"
                    )


__all__ = ["MotFieldTaskLiveOutput"]
