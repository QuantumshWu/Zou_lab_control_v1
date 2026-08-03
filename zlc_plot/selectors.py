"""Backend-neutral selector state and repeat-safe pointer gestures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
from typing import TypeAlias

from ._validation import finite_real as _finite
from ._validation import integer


def _selector_precision(span: float) -> int:
    """Decimal precision shared by native and raster selector labels."""

    gap = abs(float(span)) / 1000.0 if span else 0.01
    return max(0, -int(math.ceil(math.log10(gap))))


def _selector_number(value: float, precision: int) -> str:
    """Format one selector coordinate with the shared visible-span precision."""

    if precision <= 6 and abs(float(value)) < 1.0e9:
        return f"{float(value):.{precision}f}"
    return f"{float(value):.6g}"


@dataclass(frozen=True, slots=True)
class NumericRange:
    low: float
    high: float

    def __post_init__(self) -> None:
        low = _finite(self.low, "range low")
        high = _finite(self.high, "range high")
        if high < low:
            low, high = high, low
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def span(self) -> float:
        return self.high - self.low

    def shifted(self, delta: float) -> "NumericRange":
        delta = _finite(delta, "range shift")
        return NumericRange(self.low + delta, self.high + delta)


@dataclass(frozen=True, slots=True)
class RectangleRange:
    x: NumericRange
    y: NumericRange

    def __post_init__(self) -> None:
        if not isinstance(self.x, NumericRange) or not isinstance(self.y, NumericRange):
            raise TypeError("rectangle x and y must be NumericRange")


@dataclass(frozen=True, slots=True)
class CrosshairPoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite(self.x, "crosshair x"))
        object.__setattr__(self, "y", _finite(self.y, "crosshair y"))


SelectorValue: TypeAlias = NumericRange | RectangleRange | CrosshairPoint | float


class SelectorKind(str, Enum):
    X_RANGE = "x_range"
    AREA = "area"
    CROSSHAIR = "crosshair"
    THRESHOLD = "threshold"


class DragHandle(str, Enum):
    NEW = "new"
    BODY = "body"
    LOW = "low"
    HIGH = "high"
    LEFT = "left"
    RIGHT = "right"
    BOTTOM = "bottom"
    TOP = "top"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"


@dataclass(frozen=True, slots=True)
class SelectorState:
    kind: SelectorKind
    value: SelectorValue
    revision: int = 0
    facet_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        revision = integer(self.revision, "revision", minimum=0)
        assert revision is not None
        facet_index = integer(
            self.facet_index,
            "facet_index",
            minimum=0,
            optional=True,
        )
        _validate_value(self.kind, self.value)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "facet_index", facet_index)
        if self.kind is SelectorKind.THRESHOLD:
            object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True, slots=True)
class SelectorSnapshot:
    """One atomic selector front with at most one state of each kind."""

    committed: tuple[SelectorState, ...]
    candidate: SelectorState | None = None

    def __post_init__(self) -> None:
        committed = tuple(self.committed)
        if any(not isinstance(state, SelectorState) for state in committed):
            raise TypeError("committed must contain SelectorState values")
        by_kind = {state.kind: state for state in committed}
        if len(by_kind) != len(committed):
            raise ValueError(
                "committed selectors must contain at most one state per kind"
            )
        candidate = self.candidate
        if candidate is not None and not isinstance(candidate, SelectorState):
            raise TypeError("candidate must be SelectorState or None")
        object.__setattr__(
            self,
            "committed",
            tuple(by_kind[kind] for kind in SelectorKind if kind in by_kind),
        )

    @property
    def states(self) -> tuple[SelectorState, ...]:
        """Return the scene states after the transient candidate takes ownership."""

        by_kind = {state.kind: state for state in self.committed}
        if self.candidate is not None:
            by_kind[self.candidate.kind] = self.candidate
        return tuple(by_kind[kind] for kind in SelectorKind if kind in by_kind)


@dataclass(frozen=True, slots=True)
class GestureState:
    kind: SelectorKind
    handle: DragHandle
    origin: CrosshairPoint
    original_value: SelectorValue
    candidate_value: SelectorValue
    revision: int


def _drag_numeric_range(
    original: NumericRange,
    *,
    handle: DragHandle,
    origin: float,
    position: float,
    minimum_span: float,
    bounds: NumericRange | None = None,
) -> NumericRange:
    """Resolve one bounded NEW/BODY/LOW/HIGH numeric-range drag."""

    if not isinstance(original, NumericRange):
        raise TypeError("original must be NumericRange")
    if handle not in {
        DragHandle.NEW,
        DragHandle.BODY,
        DragHandle.LOW,
        DragHandle.HIGH,
    }:
        raise ValueError("range handle must be NEW, BODY, LOW, or HIGH")
    if bounds is not None and not isinstance(bounds, NumericRange):
        raise TypeError("bounds must be NumericRange or None")
    origin = _finite(origin, "range drag origin")
    position = _finite(position, "range drag position")
    minimum = _finite(minimum_span, "range drag minimum span")
    if minimum <= 0.0:
        raise ValueError("minimum_span must be positive")
    if bounds is not None and (bounds.span <= 0.0 or minimum > bounds.span):
        raise ValueError("bounds must contain the minimum span")
    original = _clamp_range(original, bounds)
    if handle is not DragHandle.NEW and original.span < minimum:
        raise ValueError("original is narrower than minimum_span")
    if handle is DragHandle.NEW:
        if position < origin:
            candidate = NumericRange(min(position, origin - minimum), origin)
        else:
            candidate = NumericRange(origin, max(position, origin + minimum))
        return _clamp_range(candidate, bounds)
    if handle is DragHandle.LOW:
        low = min(_clamp(position, bounds), original.high - minimum)
        if bounds is not None:
            low = max(low, bounds.low)
        return NumericRange(low, original.high)
    if handle is DragHandle.HIGH:
        high = max(_clamp(position, bounds), original.low + minimum)
        if bounds is not None:
            high = min(high, bounds.high)
        return NumericRange(original.low, high)
    return _clamp_range(original.shifted(position - origin), bounds)


def _validate_value(kind: SelectorKind, value: SelectorValue) -> None:
    if kind is SelectorKind.X_RANGE:
        if not isinstance(value, NumericRange):
            raise TypeError(f"{kind.value} requires NumericRange")
    elif kind is SelectorKind.AREA:
        if not isinstance(value, RectangleRange):
            raise TypeError("area selector requires RectangleRange")
    elif kind is SelectorKind.CROSSHAIR:
        if not isinstance(value, CrosshairPoint):
            raise TypeError("crosshair selector requires CrosshairPoint")
    elif kind is SelectorKind.THRESHOLD:
        _finite(value, "threshold")


def _stored_state(
    state: SelectorState,
    revisions: dict[SelectorKind, int],
) -> SelectorState:
    previous_revision = revisions.get(state.kind)
    revision = state.revision
    if previous_revision is not None:
        revision = max(revision, previous_revision + 1)
    return replace(state, revision=revision)


def _clamp(value: float, bounds: NumericRange | None) -> float:
    if bounds is None:
        return value
    return min(max(value, bounds.low), bounds.high)


def _clamp_range(value: NumericRange, bounds: NumericRange | None) -> NumericRange:
    if bounds is None:
        return value
    if value.span >= bounds.span:
        return bounds
    low = value.low
    high = value.high
    if low < bounds.low:
        high += bounds.low - low
        low = bounds.low
    if high > bounds.high:
        low -= high - bounds.high
        high = bounds.high
    return NumericRange(max(low, bounds.low), min(high, bounds.high))


class _SelectorController:
    """Own committed selectors and one isolated transient pointer candidate."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: dict[SelectorKind, SelectorState] = {}
        self._revisions: dict[SelectorKind, int] = {}
        self._draft: SelectorState | None = None
        self._gesture: GestureState | None = None

    @property
    def active_gesture(self) -> GestureState | None:
        with self._lock:
            return self._gesture

    def states(self) -> tuple[SelectorState, ...]:
        with self._lock:
            return tuple(
                self._states[kind]
                for kind in SelectorKind
                if kind in self._states
            )

    def snapshot(self, *, include_candidate: bool = True) -> SelectorSnapshot:
        """Capture committed and transient state under one controller lock."""

        if not isinstance(include_candidate, bool):
            raise TypeError("include_candidate must be bool")
        with self._lock:
            committed = tuple(
                self._states[kind]
                for kind in SelectorKind
                if kind in self._states
            )
            candidate = self._candidate_state_locked() if include_candidate else None
            return SelectorSnapshot(committed, candidate)

    def state(self, kind: SelectorKind) -> SelectorState:
        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        with self._lock:
            return self._states[kind]

    def install(
        self,
        state: SelectorState,
    ) -> tuple[SelectorState | None, SelectorState]:
        """Atomically replace the one committed state for ``state.kind``."""

        if not isinstance(state, SelectorState):
            raise TypeError("state must be SelectorState")
        with self._lock:
            if self._gesture is not None:
                raise RuntimeError("cannot install a selector during a pointer gesture")
            previous = self._states.get(state.kind)
            stored = _stored_state(state, self._revisions)
            self._revisions[state.kind] = stored.revision
            self._states[state.kind] = stored
        return previous, stored

    def _commit_finished(
        self,
        state: SelectorState,
    ) -> tuple[SelectorState | None, SelectorState]:
        """Promote the exact final gesture geometry without another revision."""

        if not isinstance(state, SelectorState):
            raise TypeError("state must be SelectorState")
        with self._lock:
            if self._gesture is not None or self._draft is not None:
                raise RuntimeError("cannot promote an unfinished pointer gesture")
            known = self._revisions.get(state.kind, state.revision)
            if state.revision < known:
                raise RuntimeError("finished selector revision is stale")
            previous = self._states.get(state.kind)
            self._revisions[state.kind] = state.revision
            self._states[state.kind] = state
        return previous, state

    def remove(self, kind: SelectorKind) -> SelectorState:
        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        with self._lock:
            removed = self._states.pop(kind)
            if self._gesture is not None and self._gesture.kind is kind:
                self._gesture = None
                self._draft = None
            return removed

    def _restore_removed(self, state: SelectorState) -> None:
        """Restore an exact state when its enclosing presentation rolls back."""

        if not isinstance(state, SelectorState):
            raise TypeError("state must be SelectorState")
        with self._lock:
            if state.kind in self._states:
                raise RuntimeError("cannot restore a selector kind that is present")
            self._states[state.kind] = state

    def _rollback_install(
        self,
        installed: SelectorState,
        previous: SelectorState | None,
    ) -> None:
        """Restore the selector front preceding one failed presentation."""

        if not isinstance(installed, SelectorState):
            raise TypeError("installed must be SelectorState")
        if previous is not None and not isinstance(previous, SelectorState):
            raise TypeError("previous must be SelectorState or None")
        with self._lock:
            if self._states.get(installed.kind) is not installed:
                raise RuntimeError("installed selector is no longer current")
            if previous is None:
                del self._states[installed.kind]
            else:
                if previous.kind is not installed.kind:
                    raise ValueError("previous selector kind differs from installed kind")
                self._states[installed.kind] = previous

    def pointer_down(
        self,
        kind: SelectorKind,
        x: float,
        y: float,
        *,
        handle: DragHandle = DragHandle.BODY,
        draft: SelectorState | None = None,
    ) -> None:
        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        if not isinstance(handle, DragHandle):
            raise TypeError("handle must be DragHandle")
        point = CrosshairPoint(x, y)
        with self._lock:
            if self._gesture is not None:
                self._gesture = None
                self._draft = None
            if draft is None:
                state = self._states[kind]
            else:
                if not isinstance(draft, SelectorState):
                    raise TypeError("draft must be SelectorState or None")
                if draft.kind is not kind:
                    raise ValueError("draft selector kind does not match pointer target")
                state = draft
            if draft is not None:
                self._draft = state
            self._gesture = GestureState(
                state.kind,
                handle,
                point,
                state.value,
                state.value,
                state.revision,
            )

    def pointer_move(
        self,
        x: float,
        y: float,
        *,
        x_bounds: NumericRange | None = None,
        y_bounds: NumericRange | None = None,
    ) -> SelectorState | None:
        point = CrosshairPoint(x, y)
        with self._lock:
            gesture = self._gesture
            if gesture is None:
                return None
            current = (
                self._draft
                if self._draft is not None
                else self._states[gesture.kind]
            )
            value = _dragged_value(
                current.kind,
                gesture,
                point,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
            )
            if value == gesture.candidate_value:
                return replace(
                    current,
                    value=gesture.candidate_value,
                    revision=gesture.revision,
                )
            revision = max(
                gesture.revision,
                current.revision,
                self._revisions.get(current.kind, current.revision),
            ) + 1
            self._revisions[current.kind] = revision
            self._gesture = replace(
                gesture,
                candidate_value=value,
                revision=revision,
            )
            updated = replace(current, value=value, revision=revision)
        return updated

    def finish_gesture(
        self,
        x: float,
        y: float,
        *,
        x_bounds: NumericRange | None = None,
        y_bounds: NumericRange | None = None,
    ) -> SelectorState | None:
        """Return the final transient value without committing controller state."""

        try:
            return self.pointer_move(
                x,
                y,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
            )
        finally:
            with self._lock:
                self._gesture = None
                self._draft = None

    def candidate_state(self) -> SelectorState | None:
        """Return the transient pointer value without changing committed state."""

        with self._lock:
            return self._candidate_state_locked()

    def cancel(self) -> SelectorState | None:
        try:
            with self._lock:
                gesture = self._gesture
                if gesture is None:
                    return None
                return (
                    self._draft
                    if self._draft is not None
                    else self._states[gesture.kind]
                )
        finally:
            with self._lock:
                self._gesture = None
                self._draft = None

    def lost_pointer_capture(self) -> SelectorState | None:
        return self.cancel()

    def _candidate_state_locked(self) -> SelectorState | None:
        gesture = self._gesture
        if gesture is None:
            return None
        current = (
            self._draft
            if self._draft is not None
            else self._states[gesture.kind]
        )
        return replace(
            current,
            value=gesture.candidate_value,
            revision=gesture.revision,
        )

def _dragged_value(
    kind: SelectorKind,
    gesture: GestureState,
    point: CrosshairPoint,
    *,
    x_bounds: NumericRange | None,
    y_bounds: NumericRange | None,
) -> SelectorValue:
    handle = gesture.handle
    origin = gesture.origin
    original = gesture.original_value
    if kind is SelectorKind.X_RANGE:
        assert isinstance(original, NumericRange)
        if handle is DragHandle.NEW:
            return _clamp_range(NumericRange(origin.x, point.x), x_bounds)
        if handle in {DragHandle.LOW, DragHandle.LEFT}:
            return _clamp_range(NumericRange(point.x, original.high), x_bounds)
        if handle in {DragHandle.HIGH, DragHandle.RIGHT}:
            return _clamp_range(NumericRange(original.low, point.x), x_bounds)
        return _clamp_range(original.shifted(point.x - origin.x), x_bounds)
    if kind is SelectorKind.AREA:
        assert isinstance(original, RectangleRange)
        if handle is DragHandle.NEW:
            return RectangleRange(
                _clamp_range(NumericRange(origin.x, point.x), x_bounds),
                _clamp_range(NumericRange(origin.y, point.y), y_bounds),
            )
        x = original.x
        y = original.y
        if handle is DragHandle.BODY:
            x = _clamp_range(x.shifted(point.x - origin.x), x_bounds)
            y = _clamp_range(y.shifted(point.y - origin.y), y_bounds)
        else:
            if handle in {DragHandle.LEFT, DragHandle.BOTTOM_LEFT, DragHandle.TOP_LEFT}:
                x = _clamp_range(NumericRange(point.x, x.high), x_bounds)
            if handle in {DragHandle.RIGHT, DragHandle.BOTTOM_RIGHT, DragHandle.TOP_RIGHT}:
                x = _clamp_range(NumericRange(x.low, point.x), x_bounds)
            if handle in {DragHandle.BOTTOM, DragHandle.BOTTOM_LEFT, DragHandle.BOTTOM_RIGHT}:
                y = _clamp_range(NumericRange(point.y, y.high), y_bounds)
            if handle in {DragHandle.TOP, DragHandle.TOP_LEFT, DragHandle.TOP_RIGHT}:
                y = _clamp_range(NumericRange(y.low, point.y), y_bounds)
        return RectangleRange(x, y)
    if kind is SelectorKind.CROSSHAIR:
        return CrosshairPoint(_clamp(point.x, x_bounds), _clamp(point.y, y_bounds))
    if kind is SelectorKind.THRESHOLD:
        return _clamp(point.y, y_bounds)
    raise RuntimeError(f"unsupported selector kind: {kind}")


__all__ = [
    "CrosshairPoint",
    "DragHandle",
    "GestureState",
    "NumericRange",
    "RectangleRange",
    "SelectorKind",
    "SelectorSnapshot",
    "SelectorState",
    "SelectorValue",
]
