"""Named live-signal hub: the contract between an experiment loop and live views.

An experiment logic node (real or virtual) ``publish()``-es a dict of named values once
per shot -- scalars (loading rate) or arrays (camera frame, per-site counts).  A
consumer (the task console) polls ``latest()``/``history()`` and uses ``version``
to skip work when nothing new arrived.  The hub is the ONLY shared state between
the acquisition thread and the GUI thread, so every access is lock-protected and
returns copies/stacks, never live references.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Mapping

import numpy as np

DEFAULT_HISTORY = 2048


class SignalHub:
    """Thread-safe store of named per-shot signals with bounded history."""

    def __init__(self, *, history: int = DEFAULT_HISTORY):
        self._history_len = max(1, int(history))
        self._lock = threading.RLock()
        self._signals: dict[str, deque] = {}
        self._version = 0
        self._shot = 0
        # Per-signal publish counter: lets a consumer tell "MY signal got a new
        # sample" from "the global version bumped because some OTHER logic node
        # published" -- e.g. a rolling monitor must append one point per sample
        # of its own source, not one per unrelated logic node tick.
        self._sig_version: dict[str, int] = {}

    # ------------------------------------------------------------- publish side
    def publish(self, values: Mapping[str, object], *, shot: int | None = None) -> int:
        """Record one shot's worth of named values; returns the new version.

        Every value is coerced to a float ndarray (scalars become 0-d) and COPIED,
        so the logic node may freely reuse its buffers.
        """

        with self._lock:
            self._shot = int(shot) if shot is not None else self._shot + 1
            for name, value in values.items():
                key = str(name)
                arr = np.array(value, dtype=float, copy=True)
                ring = self._signals.get(key)
                if ring is None:
                    ring = deque(maxlen=self._history_len)
                    self._signals[key] = ring
                ring.append(arr)
                self._sig_version[key] = self._sig_version.get(key, 0) + 1
            self._version += 1
            return self._version

    def clear(self) -> None:
        with self._lock:
            self._signals.clear()
            self._sig_version.clear()
            self._version += 1
            self._shot = 0

    def remove_signals(self, names) -> list[str]:
        """Drop the named signals (history + version counter) and return the ones actually removed.

        The TARGETED opposite of :meth:`clear`: when a logic node is REMOVED from the console, its
        published signals are stale and must leave the hub so they stop cluttering every picker
        ("多余 signal" that accumulate run after run).  STOPPING a node does NOT call this -- a stopped
        node's signals deliberately linger (so a finished scan stays plottable and a panel can be wired
        before the next run); only REMOVING the node purges them.  A no-op for names that are not present
        (idempotent); bumps the global version once if anything was removed so consumers refresh."""
        removed: list[str] = []
        with self._lock:
            for name in (str(n) for n in (names or [])):
                if name in self._signals:
                    del self._signals[name]
                    self._sig_version.pop(name, None)
                    removed.append(name)
            if removed:
                self._version += 1
        return removed

    def signal_versions(self) -> dict[str, int]:
        """``{name: publish_count}`` snapshot -- one counter per signal, bumped
        each time that name is published.  A consumer compares a name's counter
        across ticks to detect a NEW sample of THAT signal (vs. a global version
        bump caused by an unrelated logic node)."""
        with self._lock:
            return dict(self._sig_version)

    # ------------------------------------------------------------- consumer side
    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def shot(self) -> int:
        with self._lock:
            return self._shot

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._signals)

    def latest(self, name: str):
        """Most recent value of ``name`` (float scalar for 0-d, else ndarray copy)."""

        with self._lock:
            ring = self._signals.get(str(name))
            if not ring:
                raise KeyError(f"no signal named {name!r}; available: {sorted(self._signals)}")
            value = ring[-1]
        return float(value) if value.ndim == 0 else value.copy()

    def history(self, name: str, n: int | None = None) -> np.ndarray:
        """Last ``n`` shots of ``name`` stacked on axis 0 (shape ``(shots, *value_shape)``).

        Shorter-than-``n`` history returns what exists; shapes that changed over time
        keep only the run of MOST RECENT shots with the current shape (a logic node that
        reconfigured mid-run must not corrupt the stack).
        """

        with self._lock:
            ring = self._signals.get(str(name))
            if not ring:
                raise KeyError(f"no signal named {name!r}; available: {sorted(self._signals)}")
            items = list(ring) if n is None else list(ring)[-max(1, int(n)):]
        shape = items[-1].shape
        usable = []
        for item in reversed(items):
            if item.shape != shape:
                break
            usable.append(item)
        return np.stack(list(reversed(usable)), axis=0)

    def snapshot_latest(self) -> dict[str, object]:
        """One consistent {name: latest value} mapping (the expression namespace)."""

        with self._lock:
            names = list(self._signals)
        out: dict[str, object] = {}
        for name in names:
            try:
                out[name] = self.latest(name)
            except KeyError:  # pragma: no cover - racy clear() between list and read
                continue
        return out


__all__ = ["SignalHub", "DEFAULT_HISTORY"]
