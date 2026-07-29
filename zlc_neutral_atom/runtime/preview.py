"""Provisional Dataset display attachment contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from zlc_data import BlockId
from zlc_storage import sha256_text

from ._failure import safe_error_summary
from .dataset import ExactDatasetPreviewReader, FrozenDatasetEdge


@runtime_checkable
class LiveDatasetViewSpec(Protocol):
    """Domain-neutral identity and edge required by a live UI attachment."""

    block_id: BlockId
    dataset_edge: FrozenDatasetEdge


@dataclass(frozen=True)
class ExactDatasetPreviewSpec:
    """Display branch over an exact builder's current revision."""

    source_schema_fingerprint: str

    def __post_init__(self) -> None:
        sha256_text(self.source_schema_fingerprint, "source_schema_fingerprint")


class ExactDatasetPreviewPort(Protocol):
    """Runtime-owned provisional reader attachment; never artifact authority."""

    @property
    def spec(self) -> ExactDatasetPreviewSpec: ...

    @property
    def terminal(self) -> bool: ...

    def bind(
        self,
        reader: ExactDatasetPreviewReader,
        *,
        run_id: str,
        causation_domain_id: str,
    ) -> None: ...

    def updated(self) -> None: ...

    def fail(self, message: str) -> None: ...

    def source_terminal(self) -> None: ...


class FailureAwarePreviewPort(Protocol):
    """Small failure surface shared by independent provisional views."""

    @property
    def terminal(self) -> bool: ...

    def fail(self, message: str) -> None: ...


def notify_preview_failure(
    preview: FailureAwarePreviewPort | None,
    error: BaseException,
) -> None:
    """Best-effort terminal notification; display failure never masks the run."""

    if preview is None:
        return
    try:
        if preview.terminal:
            return
        preview.fail(safe_error_summary(error))
    except BaseException:
        pass


__all__ = [
    "ExactDatasetPreviewPort",
    "ExactDatasetPreviewSpec",
    "FailureAwarePreviewPort",
    "LiveDatasetViewSpec",
    "notify_preview_failure",
]
