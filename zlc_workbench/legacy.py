"""Narrow, fail-closed migration seams for the legacy TaskConsole island."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from typing import Callable

from zlc_storage import canonical_text as _text


def _normalize_human_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized


class CatalogRoute(Enum):
    LEGACY = "legacy"
    TARGET = "target"
    HIDDEN = "hidden"


@dataclass(frozen=True)
class CatalogEntry:
    use_case_id: str
    route: CatalogRoute
    hidden_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "use_case_id", _text(self.use_case_id, "use_case_id")
        )
        if not isinstance(self.route, CatalogRoute):
            raise TypeError("route must be CatalogRoute")
        if self.route is CatalogRoute.HIDDEN:
            object.__setattr__(
                self,
                "hidden_reason",
                _normalize_human_text(self.hidden_reason, "hidden_reason"),
            )
        elif self.hidden_reason is not None:
            raise ValueError("hidden_reason is only valid for HIDDEN routes")


class CatalogRouter:
    """Exactly one visible implementation per use case during migration."""

    def __init__(
        self,
        expected_use_case_ids: tuple[str, ...],
        entries: tuple[CatalogEntry, ...],
    ) -> None:
        expected = tuple(
            _text(value, "expected use_case_id") for value in expected_use_case_ids
        )
        if len(set(expected)) != len(expected):
            raise ValueError("expected use-case ids must be unique")
        entries = tuple(entries)
        if any(not isinstance(entry, CatalogEntry) for entry in entries):
            raise TypeError("entries must contain CatalogEntry values")
        ids = tuple(entry.use_case_id for entry in entries)
        if len(set(ids)) != len(ids):
            raise ValueError("each use case must have exactly one catalog route")
        missing = set(expected) - set(ids)
        extra = set(ids) - set(expected)
        if missing or extra:
            detail = []
            if missing:
                detail.append("missing=" + ",".join(sorted(missing)))
            if extra:
                detail.append("extra=" + ",".join(sorted(extra)))
            raise ValueError("catalog routes must exactly cover the expected catalog: " + "; ".join(detail))
        self._entries = {entry.use_case_id: entry for entry in entries}

    def resolve(self, use_case_id: str) -> CatalogEntry:
        key = _text(use_case_id, "use_case_id")
        try:
            return self._entries[key]
        except KeyError as exc:
            raise KeyError(f"use case {key!r} has no explicit catalog route") from exc


class LegacyHandoffTimeout(RuntimeError):
    pass


class SerializedLegacyAggBridge:
    """Fail-closed facade over the current shared-Figure RenderLoop island."""

    def __init__(self, render_loop: object) -> None:
        required = ("submit", "barrier", "stop")
        if not all(callable(getattr(render_loop, name, None)) for name in required):
            raise TypeError("render_loop must provide submit(), barrier(), and stop()")
        self._render_loop = render_loop
        self._owner_thread = threading.get_ident()
        self._poisoned = False
        self._closed = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def closed(self) -> bool:
        return self._closed

    def submit(self, job: Callable[[], object]) -> bool:
        self._require_owner()
        self._ensure_usable()
        if not callable(job):
            raise TypeError("job must be callable")
        return bool(self._render_loop.submit(job))

    def settle(self, timeout: float = 5.0) -> None:
        self._require_owner()
        self._ensure_usable()
        timeout = self._timeout(timeout)
        try:
            settled = self._render_loop.barrier(timeout)
        except BaseException as exc:
            self._poisoned = True
            raise LegacyHandoffTimeout(
                "legacy Agg ownership handoff failed; figure access remains forbidden"
            ) from exc
        if not settled:
            self._poisoned = True
            raise LegacyHandoffTimeout(
                "legacy Agg ownership handoff timed out; figure access remains forbidden"
            )

    def close(self, timeout: float = 5.0) -> None:
        self._require_owner()
        timeout = self._timeout(timeout)
        if self._closed:
            return
        if not self._poisoned:
            try:
                self.settle(timeout)
            except LegacyHandoffTimeout:
                pass
        try:
            terminated = self._render_loop.stop(timeout)
        except BaseException as exc:
            self._poisoned = True
            raise LegacyHandoffTimeout(
                "legacy render worker stop failed; Figure teardown remains forbidden"
            ) from exc
        if terminated is not True:
            self._poisoned = True
            raise LegacyHandoffTimeout(
                "legacy render worker did not terminate; Figure teardown remains forbidden"
            )
        self._closed = True

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("legacy Agg handoff is GUI-owner-thread affine")

    def _ensure_usable(self) -> None:
        if self._closed:
            raise RuntimeError("legacy Agg bridge is closed")
        if self._poisoned:
            raise RuntimeError("legacy Agg bridge is poisoned after failed handoff")

    @staticmethod
    def _timeout(value: object) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("timeout must be positive")
        return float(value)


__all__ = [
    "CatalogEntry",
    "CatalogRoute",
    "CatalogRouter",
    "LegacyHandoffTimeout",
    "SerializedLegacyAggBridge",
]
