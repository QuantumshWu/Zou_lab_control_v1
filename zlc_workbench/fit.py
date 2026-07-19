"""Headless process-local authority for one interactive fit draft.

The Qt host may display ``FitDraftResult`` and request commands, but it never
receives the ``FitExecution`` capability that can publish an artifact.  This
owner accepts the one neutral-owned Fit execution capability; it is not an
analysis registry, workflow engine, executor, or persistence layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading

from zlc_data import FitCancelled, FitResultBatch, FitSpec
from zlc_neutral_atom.artifacts import FitExecution
from zlc_neutral_atom.fit_reference import FitResultArtifactRef


@dataclass(frozen=True, slots=True, eq=False)
class FitDraftResult:
    """Immutable, non-saving result view admitted by one draft authority."""

    generation: int
    result: FitResultBatch

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError("fit draft generation must be int")
        if self.generation <= 0:
            raise ValueError("fit draft generation must be positive")
        if not isinstance(self.result, FitResultBatch):
            raise TypeError("fit draft result must be FitResultBatch")


class FitDraftAuthority:
    """Own exactly one unsaved ``FitExecution`` behind non-saving commands."""

    def __init__(
        self,
        execute_fit: Callable[
            [FitSpec, Callable[[], bool], float],
            FitExecution,
        ],
        save_fit: Callable[[FitExecution], FitResultArtifactRef],
    ) -> None:
        if not callable(execute_fit):
            raise TypeError("execute_fit must be callable")
        if not callable(save_fit):
            raise TypeError("save_fit must be callable")
        self._lock = threading.Lock()
        self._execute_fit = execute_fit
        self._save_fit = save_fit
        self._generation = 0
        self._draft: FitDraftResult | None = None
        self._execution: FitExecution | None = None
        self._saving_generation: int | None = None
        self._closed = False

    def execute(
        self,
        spec: FitSpec,
        cancel_check: Callable[[], bool],
        deadline_monotonic: float,
    ) -> FitDraftResult:
        if not isinstance(spec, FitSpec):
            raise TypeError("spec must be FitSpec")
        if not callable(cancel_check):
            raise TypeError("cancel_check must be callable")
        with self._lock:
            if self._closed:
                raise FitCancelled("fit draft authority is closed")
            if self._execution is not None or self._saving_generation is not None:
                raise RuntimeError("discard the current fit draft before executing")
            execute_fit = self._execute_fit
        execution = execute_fit(spec, cancel_check, deadline_monotonic)
        if not isinstance(execution, FitExecution):
            raise TypeError("fit executor must return FitExecution")
        if cancel_check():
            raise FitCancelled("fit was cancelled before draft admission")
        with self._lock:
            if self._closed or cancel_check():
                raise FitCancelled("fit draft authority closed before admission")
            if self._execution is not None or self._saving_generation is not None:
                raise RuntimeError("fit draft authority changed during execution")
            self._generation += 1
            draft = FitDraftResult(self._generation, execution.result)
            self._draft = draft
            self._execution = execution
            return draft

    def discard(self, draft: FitDraftResult) -> bool:
        """Drop an unsaved draft; a save already in progress remains atomic."""

        if not isinstance(draft, FitDraftResult):
            raise TypeError("draft must be FitDraftResult")
        with self._lock:
            if self._draft is not draft or self._execution is None:
                return False
            if self._saving_generation == draft.generation:
                return False
            self._draft = None
            self._execution = None
            return True

    def save(self, draft: FitDraftResult) -> FitResultArtifactRef:
        """Publish the exact current draft, retaining it if publication fails."""

        with self._lock:
            execution = self._require_current_locked(draft)
            if self._saving_generation is not None:
                raise RuntimeError("fit draft is already being saved")
            self._saving_generation = draft.generation
            save_fit = self._save_fit
        try:
            if save_fit is None:
                raise FitCancelled("fit draft authority is closed")
            reference = save_fit(execution)
            if not isinstance(reference, FitResultArtifactRef):
                raise TypeError("fit saver returned an invalid reference")
        except BaseException:
            with self._lock:
                if self._saving_generation == draft.generation:
                    self._saving_generation = None
                    if self._closed:
                        self._draft = None
                        self._execution = None
                        self._save_fit = None
            raise
        with self._lock:
            if self._saving_generation != draft.generation:
                raise RuntimeError("fit save authority changed during publication")
            self._saving_generation = None
            self._draft = None
            self._execution = None
            if self._closed:
                self._save_fit = None
        return reference

    def close(self) -> None:
        """Revoke future drafts without interrupting an admitted atomic save."""

        with self._lock:
            self._closed = True
            self._execute_fit = None
            if self._saving_generation is None:
                self._draft = None
                self._execution = None
                self._save_fit = None

    def _require_current_locked(self, draft: FitDraftResult) -> FitExecution:
        if not isinstance(draft, FitDraftResult):
            raise TypeError("draft must be FitDraftResult")
        if self._draft is not draft or self._execution is None:
            raise RuntimeError("fit draft is stale or no longer admitted")
        return self._execution


__all__ = ["FitDraftAuthority", "FitDraftResult"]
