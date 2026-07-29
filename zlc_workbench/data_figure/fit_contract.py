"""Workbench-only execution and persistence capabilities for Figure Fit."""

from __future__ import annotations

from dataclasses import dataclass
import math

from zlc_data import FitResultBatch, Selection


DEFAULT_FIT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class FitWorkbenchBindings:
    """Closed application capabilities consumed by the optional Fit surface."""

    prepare: object
    execute: object
    result: object
    save: object
    reload: object
    selected_model: str | None = None
    initial_selection: Selection | None = None
    open_fit: bool = False
    timeout_seconds: float = DEFAULT_FIT_TIMEOUT_SECONDS
    save_requires_path: bool = False
    initial_save_path: object | None = None

    def __post_init__(self) -> None:
        for name in ("prepare", "execute", "result", "save", "reload"):
            if not callable(getattr(self, name)):
                raise TypeError(f"fit {name} capability must be callable")
        selected = self.selected_model
        if selected is not None and (
            not isinstance(selected, str) or not selected.strip()
        ):
            raise ValueError("selected_model must be non-empty text or None")
        if self.initial_selection is not None and not isinstance(
            self.initial_selection,
            Selection,
        ):
            raise TypeError("initial_selection must be Selection or None")
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("fit timeout_seconds must be finite and positive")
        object.__setattr__(self, "timeout_seconds", timeout)
        if not isinstance(self.save_requires_path, bool):
            raise TypeError("save_requires_path must be bool")
        path = self.initial_save_path
        if path is not None:
            from pathlib import Path

            object.__setattr__(self, "initial_save_path", Path(path))
        if not self.save_requires_path and path is not None:
            raise ValueError(
                "an initial Fit save path requires save_requires_path=True"
            )


@dataclass(frozen=True, slots=True)
class FitSaveReceipt:
    """One admitted persistence result shared by artifact and archive saves."""

    handle: object
    identity: str
    summary: str
    reloaded_result: FitResultBatch | None = None
    artifact_reference: object | None = None

    def __post_init__(self) -> None:
        if self.handle is None:
            raise TypeError("Fit save receipt requires one persistence handle")
        for name in ("identity", "summary"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Fit save receipt {name} must be non-empty text")
        if self.reloaded_result is not None and not isinstance(
            self.reloaded_result,
            FitResultBatch,
        ):
            raise TypeError(
                "Fit save receipt reloaded_result must be FitResultBatch or None"
            )


__all__ = [
    "DEFAULT_FIT_TIMEOUT_SECONDS",
    "FitSaveReceipt",
    "FitWorkbenchBindings",
]
