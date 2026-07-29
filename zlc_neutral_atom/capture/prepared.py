"""Shared one-shot preparation and preview surface for exact camera capture."""

from __future__ import annotations

import threading
from typing import Callable
from uuid import uuid4

from zlc_data import (
    BlockId,
    DatasetSchema,
    READOUT_EVENT,
)
from .artifact import (
    CaptureRepository,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.capture.pipeline import (
    CapturePreviewPort,
    CapturePreviewSpec,
    MinimalPipelineSpec,
)
from zlc_neutral_atom.runtime.preview import notify_preview_failure
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_neutral_atom.capture.triggered import TriggeredCaptureSpec
from zlc_storage import canonical_text


class PreparedExactCapture:
    """Shared one-shot command boundary for exact camera capture products."""

    __slots__ = (
        "_capture",
        "_descriptor",
        "_lock",
        "_one_shot_name",
        "_pipeline",
        "_preview_block_id",
        "_preview_edge",
        "_preview_schema",
        "_repository",
        "_start_run",
        "_started",
    )

    def __init__(
        self,
        capture: MinimalPipelineSpec | TriggeredCaptureSpec,
        repository: CaptureRepository,
        start_run: Callable[[RunPlan], RunHandle],
        descriptor: object,
        *,
        one_shot_name: str,
    ) -> None:
        if not isinstance(capture, (MinimalPipelineSpec, TriggeredCaptureSpec)):
            raise TypeError("capture must be an exact camera pipeline spec")
        if type(repository) is not CaptureRepository:
            raise TypeError("repository must be CaptureRepository")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        self._capture = capture
        self._pipeline = (
            capture.capture if isinstance(capture, TriggeredCaptureSpec) else capture
        )
        self._repository = repository
        self._start_run = start_run
        self._descriptor = descriptor
        self._one_shot_name = canonical_text(one_shot_name, "one_shot_name")
        self._preview_block_id = BlockId(f"capture-preview-{uuid4().hex}")
        self._preview_edge = CapturePreviewSpec.dataset_edge_for_capture(
            self._pipeline
        )
        self._lock = threading.Lock()
        self._preview_schema: DatasetSchema | None = None
        self._started = False

    @property
    def descriptor(self) -> object:
        return self._descriptor

    @property
    def preview_schema(self) -> DatasetSchema:
        with self._lock:
            if self._preview_schema is not None:
                return self._preview_schema
            schema = self._pipeline.capture.capture_contract.dataset_schema
            readout_columns = tuple(
                column
                for column in schema.point_table.columns
                if column.role == READOUT_EVENT
            )
            if (
                len(readout_columns) != 1
                or len(schema.point_table.columns) != 1
                or readout_columns[0].values
                != tuple(range(schema.point_table.row_count))
            ):
                raise ValueError(
                    "finite Camera preview requires one explicit READOUT_EVENT "
                    "axis and no scan-point multiplexing"
                )
            self._preview_schema = self._preview_edge.schema
            return self._preview_schema

    def start(self, *, lifecycle_owner: object | None = None) -> RunHandle:
        self._claim_start()
        plan = compile_capture_artifact_pipeline(
            self._capture,
            self._repository,
        )
        return self._start_run(
            plan.with_lifecycle(
                owner=self if lifecycle_owner is None else lifecycle_owner,
                preemptible=False,
            )
        )

    def start_with_preview(
        self,
        *,
        factory: Callable[[CapturePreviewSpec], CapturePreviewPort],
        source_ordinals: tuple[int, ...] | None = None,
        lifecycle_owner: object | None = None,
    ) -> RunHandle:
        """Start once, optionally publishing only named physical frame ordinals."""

        if not callable(factory):
            raise TypeError("factory must be callable")
        self.preview_schema
        self._claim_start()
        preview_spec = CapturePreviewSpec(
            self._preview_block_id,
            self._preview_edge,
            source_ordinals,
        )
        preview = factory(preview_spec)
        plan = compile_capture_artifact_pipeline(
            self._capture,
            self._repository,
            preview=preview,
        )
        try:
            return self._start_run(
                plan.with_lifecycle(
                    owner=self if lifecycle_owner is None else lifecycle_owner,
                    preemptible=False,
                )
            )
        except BaseException as error:
            notify_preview_failure(preview, error)
            raise

    def _claim_start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError(f"{self._one_shot_name} is one-shot")
            self._started = True


__all__ = ["PreparedExactCapture"]
