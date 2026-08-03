"""Immutable display revisions and atomic parameter updates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from .parameters import FrozenParameters, ParameterSchema, RenderEffect


@dataclass(frozen=True, slots=True)
class DisplayState:
    """One immutable, monotonically numbered display configuration."""

    revision: int
    values: FrozenParameters
    changed_names: frozenset[str]
    effects: RenderEffect

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("display revision must be an integer")
        if self.revision < 0:
            raise ValueError("display revision must be non-negative")
        if not isinstance(self.values, FrozenParameters):
            object.__setattr__(self, "values", FrozenParameters(self.values))
        if not isinstance(self.changed_names, frozenset):
            object.__setattr__(self, "changed_names", frozenset(self.changed_names))
        if not all(isinstance(name, str) for name in self.changed_names):
            raise TypeError("changed_names must contain strings")
        if not isinstance(self.effects, RenderEffect):
            raise TypeError("effects must be RenderEffect")

    def __getitem__(self, name: str) -> Any:
        return self.values[name]


class DisplayStateStore:
    """Thread-safe owner of the current :class:`DisplayState`.

    A prepared transition replaces the complete immutable state in one locked
    operation, so render workers can safely retain an earlier revision while
    the UI continues editing.
    """

    __slots__ = ("_schema", "_lock", "_state")

    def __init__(
        self,
        schema: ParameterSchema,
        initial: Mapping[str, object] | None = None,
        *,
        initial_revision: int = 0,
        initial_effects: RenderEffect = RenderEffect.LAYOUT,
    ) -> None:
        if not isinstance(schema, ParameterSchema):
            raise TypeError("schema must be ParameterSchema")
        if isinstance(initial_revision, bool) or not isinstance(initial_revision, int):
            raise TypeError("initial_revision must be an integer")
        if initial_revision < 0:
            raise ValueError("initial_revision must be non-negative")
        if not isinstance(initial_effects, RenderEffect):
            raise TypeError("initial_effects must be RenderEffect")
        values = schema.initial_values(initial)
        self._schema = schema
        self._lock = RLock()
        self._state = DisplayState(
            revision=initial_revision,
            values=values,
            changed_names=frozenset(schema.names),
            effects=initial_effects,
        )

    @property
    def state(self) -> DisplayState:
        with self._lock:
            return self._state

    def _commit_prepared(
        self,
        previous: DisplayState,
        candidate: FrozenParameters,
    ) -> DisplayState:
        """Atomically commit a transition already prepared by this store's schema."""

        with self._lock:
            if self._state is not previous:
                raise RuntimeError("display state changed before prepared commit")
            changed = frozenset(
                name
                for name in self._schema.names
                if previous.values[name] != candidate[name]
            )
            if not changed:
                return previous
            self._state = DisplayState(
                revision=previous.revision + 1,
                values=candidate,
                changed_names=changed,
                effects=self._schema.effects_for(changed),
            )
            return self._state

    def _restore_prepared(
        self,
        current: DisplayState,
        previous: DisplayState,
    ) -> None:
        """Roll back one unpresented transition under compare-and-swap."""

        if not isinstance(current, DisplayState) or not isinstance(
            previous, DisplayState
        ):
            raise TypeError("display rollback requires DisplayState values")
        with self._lock:
            if self._state is not current:
                raise RuntimeError("display state changed before rollback")
            self._state = previous


__all__ = [
    "DisplayState",
]
