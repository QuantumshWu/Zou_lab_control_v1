"""Read-only cooperative cancellation observed by one runtime operation."""

from __future__ import annotations

import threading

from zlc_storage import canonical_text


class CancellationRequested(RuntimeError):
    """Raised at a cooperative cancellation checkpoint."""

    def __init__(self, reason: str | None = None) -> None:
        super().__init__("operation cancellation requested" if reason is None else reason)
        self.reason = reason


class _CancellationState:
    __slots__ = ("event", "lock", "reason")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.reason: str | None = None


_TOKEN_MINT = object()


class CancellationToken:
    """Read-only view of a monotonic cancellation source.

    A token deliberately has no ``request`` method.  Code receiving a token may
    observe its owner's decision, but cannot steal cancellation ownership or
    suppress the owner's hardware-sealing transition by requesting first.
    """

    __slots__ = ("_state",)

    def __init__(self, mint: object, state: _CancellationState) -> None:
        if mint is not _TOKEN_MINT:
            raise PermissionError("CancellationToken is minted by its cancellation owner")
        self._state = state

    @property
    def is_cancelled(self) -> bool:
        return self._state.event.is_set()

    @property
    def reason(self) -> str | None:
        with self._state.lock:
            return self._state.reason

    def checkpoint(self) -> None:
        if self._state.event.is_set():
            raise CancellationRequested(self.reason)

    def wait_requested(self) -> None:
        """Block without polling, then raise the owner's cancellation signal."""

        self._state.event.wait()
        self.checkpoint()


class _CancellationSource:
    """Private write authority owned by a RunHandle or worker."""

    __slots__ = ("_state", "token")

    def __init__(self) -> None:
        state = _CancellationState()
        self._state = state
        self.token = CancellationToken(_TOKEN_MINT, state)

    def request(self, reason: str | None = None) -> bool:
        if reason is not None:
            canonical_text(reason, "cancellation reason")
        state = self._state
        with state.lock:
            if state.event.is_set():
                return False
            state.reason = reason
            state.event.set()
            return True
