"""Bounded latest-wins control commands with exact terminal acknowledgements.

The topic is deliberately smaller than an event stream.  It owns at most one
claimed command and one unclaimed replacement, and it creates no worker thread:
the component that owns the controlled resource explicitly claims and
acknowledges commands through :class:`ControlConsumer`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, TypeVar

from zlc_storage import canonical_text as _canonical_text
from zlc_storage import finite_real, positive_integer as _positive_int


ControlT = TypeVar("ControlT")
_MINT_TOKEN = object()


class ControlAckStatus(str, Enum):
    """The complete acknowledgement state vocabulary for one revision."""

    ACCEPTED = "ACCEPTED"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    TERMINATED = "TERMINATED"

    @property
    def terminal(self) -> bool:
        return self is not ControlAckStatus.ACCEPTED


@dataclass(frozen=True, slots=True)
class ControlAck:
    """One immutable acknowledgement snapshot for a published revision."""

    revision: int
    status: ControlAckStatus
    reason: str | None = None
    superseded_by: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", _positive_int(self.revision, "control revision"))
        if not isinstance(self.status, ControlAckStatus):
            raise TypeError("control acknowledgement status must be ControlAckStatus")

        if self.status in (ControlAckStatus.REJECTED, ControlAckStatus.TERMINATED):
            reason = _canonical_text(self.reason, "control acknowledgement reason")
            object.__setattr__(self, "reason", reason)
        elif self.reason is not None:
            raise ValueError(f"{self.status.value} acknowledgement cannot contain a reason")

        if self.status is ControlAckStatus.SUPERSEDED:
            successor = _positive_int(
                self.superseded_by,
                "superseding control revision",
            )
            if successor <= self.revision:
                raise ValueError("superseding control revision must be newer than the superseded revision")
            object.__setattr__(self, "superseded_by", successor)
        elif self.superseded_by is not None:
            raise ValueError(
                f"{self.status.value} acknowledgement cannot contain superseded_by"
            )

    @property
    def terminal(self) -> bool:
        return self.status.terminal


class ControlReceipt:
    """Thread-safe observation and terminal wait handle for one revision."""

    __slots__ = ("_ack", "_condition")

    def __init__(self, authority: object, revision: int) -> None:
        if authority is not _MINT_TOKEN:
            raise PermissionError("ControlReceipt can only be minted by create_control_topic")
        self._ack = ControlAck(revision, ControlAckStatus.ACCEPTED)
        self._condition = threading.Condition(threading.Lock())

    def snapshot(self) -> ControlAck:
        with self._condition:
            return self._ack

    def wait(self, timeout: float | None = None) -> ControlAck:
        timeout = (
            None
            if timeout is None
            else finite_real(timeout, "control receipt wait timeout", minimum=0.0)
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._ack.terminal:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError(
                        f"control revision {self._ack.revision} has not reached terminal state"
                    )
                self._condition.wait(remaining)
            return self._ack

    def _finish(self, authority: object, ack: ControlAck) -> None:
        if authority is not _MINT_TOKEN:
            raise PermissionError("only the owning control topic may finish a receipt")
        if not isinstance(ack, ControlAck):
            raise TypeError("terminal acknowledgement must be ControlAck")
        if not ack.terminal:
            raise ValueError("receipt can only transition to a terminal acknowledgement")
        with self._condition:
            if ack.revision != self._ack.revision:
                raise ValueError("terminal acknowledgement revision does not match receipt")
            if self._ack.terminal:
                raise RuntimeError(
                    f"control revision {self._ack.revision} is already terminal"
                )
            self._ack = ack
            self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class ControlCommand(Generic[ControlT]):
    """An owned snapshot delivered only to the topic's paired consumer."""

    revision: int
    value: ControlT
    _identity: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", _positive_int(self.revision, "control revision"))
        if self._identity is None:
            raise TypeError("control command identity cannot be None")


@dataclass(frozen=True, slots=True)
class _ControlEntry(Generic[ControlT]):
    command: ControlCommand[ControlT]
    receipt: ControlReceipt


class _ControlCore(Generic[ControlT]):
    """The shared two-slot state machine; all methods require its lock."""

    __slots__ = (
        "_identity",
        "_inflight",
        "_lock",
        "_next_revision",
        "_pending",
        "_terminated",
    )

    def __init__(self) -> None:
        self._identity = object()
        self._inflight: _ControlEntry[ControlT] | None = None
        self._lock = threading.RLock()
        self._next_revision = 1
        self._pending: _ControlEntry[ControlT] | None = None
        self._terminated = False

    def publish_snapshot(self, value: ControlT) -> ControlReceipt:
        with self._lock:
            if self._terminated:
                raise RuntimeError("control topic is terminated")
            revision = self._next_revision
            self._next_revision += 1
            command = ControlCommand(revision, value, self._identity)
            receipt = ControlReceipt(_MINT_TOKEN, revision)
            replacement = _ControlEntry(command, receipt)

            previous = self._pending
            if previous is not None:
                previous.receipt._finish(
                    _MINT_TOKEN,
                    ControlAck(
                        previous.command.revision,
                        ControlAckStatus.SUPERSEDED,
                        superseded_by=revision,
                    ),
                )
            self._pending = replacement
            return receipt

    def take_latest(self) -> ControlCommand[ControlT] | None:
        with self._lock:
            if self._terminated or self._inflight is not None or self._pending is None:
                return None
            self._inflight = self._pending
            self._pending = None
            return self._inflight.command

    def applied(self, command: ControlCommand[ControlT]) -> None:
        with self._lock:
            entry = self._require_inflight(command)
            entry.receipt._finish(
                _MINT_TOKEN,
                ControlAck(command.revision, ControlAckStatus.APPLIED),
            )
            self._inflight = None

    def rejected(self, command: ControlCommand[ControlT], reason: str) -> None:
        with self._lock:
            entry = self._require_inflight(command)
            ack = ControlAck(command.revision, ControlAckStatus.REJECTED, reason=reason)
            entry.receipt._finish(_MINT_TOKEN, ack)
            self._inflight = None

    def terminate(self, reason: str) -> None:
        reason = _canonical_text(reason, "control termination reason")
        with self._lock:
            if self._terminated:
                raise RuntimeError("control topic is already terminated")
            entries = tuple(
                entry for entry in (self._inflight, self._pending) if entry is not None
            )
            for entry in entries:
                entry.receipt._finish(
                    _MINT_TOKEN,
                    ControlAck(
                        entry.command.revision,
                        ControlAckStatus.TERMINATED,
                        reason=reason,
                    ),
                )
            self._inflight = None
            self._pending = None
            self._terminated = True

    def _require_inflight(
        self,
        command: ControlCommand[ControlT],
    ) -> _ControlEntry[ControlT]:
        if not isinstance(command, ControlCommand):
            raise TypeError("control acknowledgement requires ControlCommand")
        if command._identity is not self._identity:
            raise PermissionError("control command belongs to a different topic")
        entry = self._inflight
        if entry is None:
            raise RuntimeError("control topic has no in-flight command to acknowledge")
        if command is not entry.command:
            raise RuntimeError("control command is not the current in-flight command")
        return entry


class ControlTopic(Generic[ControlT]):
    """Publisher side of a bounded latest-wins control topic."""

    __slots__ = ("_core", "_snapshot")

    def __init__(
        self,
        authority: object,
        core: _ControlCore[ControlT],
        snapshot: Callable[[ControlT], ControlT],
    ) -> None:
        if authority is not _MINT_TOKEN:
            raise PermissionError("ControlTopic can only be minted by create_control_topic")
        self._core = core
        self._snapshot = snapshot

    def publish(self, value: ControlT) -> ControlReceipt:
        # Snapshot before entering the core, including the terminated check.  A failed
        # snapshot accepts no revision and cannot supersede an already-pending command.
        owned = self._snapshot(value)
        return self._core.publish_snapshot(owned)


class ControlConsumer(Generic[ControlT]):
    """Single resource-owner side of a :class:`ControlTopic`."""

    __slots__ = ("_core",)

    def __init__(self, authority: object, core: _ControlCore[ControlT]) -> None:
        if authority is not _MINT_TOKEN:
            raise PermissionError("ControlConsumer can only be minted by create_control_topic")
        self._core = core

    def take_latest(self) -> ControlCommand[ControlT] | None:
        return self._core.take_latest()

    def applied(self, command: ControlCommand[ControlT]) -> None:
        self._core.applied(command)

    def rejected(self, command: ControlCommand[ControlT], reason: str) -> None:
        self._core.rejected(command, reason)

    def terminate(self, reason: str) -> None:
        self._core.terminate(reason)


def create_control_topic(
    snapshot_callable: Callable[[ControlT], ControlT],
) -> tuple[ControlTopic[ControlT], ControlConsumer[ControlT]]:
    """Mint the only publisher/consumer pair for one bounded control core."""

    if not callable(snapshot_callable):
        raise TypeError("control snapshot_callable must be callable")
    core: _ControlCore[ControlT] = _ControlCore()
    return (
        ControlTopic(_MINT_TOKEN, core, snapshot_callable),
        ControlConsumer(_MINT_TOKEN, core),
    )


__all__ = [
    "ControlAck",
    "ControlAckStatus",
    "ControlCommand",
    "ControlConsumer",
    "ControlReceipt",
    "ControlTopic",
    "create_control_topic",
]
