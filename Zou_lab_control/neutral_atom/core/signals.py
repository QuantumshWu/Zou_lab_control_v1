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

#: A sample that belongs to NO acquisition shot -- a free-running scalar (loading rate) or a static aux
#: (centers / thresholds).  It never groups into a coherent display shot, so a panel bound to it is shown
#: best-effort (latest) and never blocks the shot clock (#shot-clock).
NO_LINEAGE = -1


class SignalHub:
    """Thread-safe store of named per-shot signals with bounded history.

    Each sample also carries a SOURCE-SHOT provenance id (the acquisition it derives from): an acquiring
    node mints a fresh id per real shot (:meth:`next_source_shot`); a reactive processor INHERITS the id
    of the frame it consumed, so a derived signal (``frame_judged``/``occupied``) and the ``frame`` it was
    computed from share ONE id.  A consumer then assembles a coherent cross-producer display shot via
    :meth:`snapshot_at` instead of mixing the latest of each independently-threaded producer (#shot-clock).
    """

    def __init__(self, *, history: int = DEFAULT_HISTORY):
        self._history_len = max(1, int(history))
        self._lock = threading.RLock()
        self._signals: dict[str, deque] = {}
        # Parallel per-name deque[int] of source-shot provenance ids, kept in LOCKSTEP with _signals (same
        # key set, same maxlen, appended together under the lock) so value[i] was acquired at src[i].
        self._src: dict[str, deque] = {}
        self._version = 0
        self._shot = 0
        # The SOURCE-shot counter (lineage clock), DISTINCT from _shot (a publish counter) and _version (a
        # global bump): acquiring nodes mint from it so every signal of one physical shot shares one id.
        self._source_shot = 0
        # Per-signal publish counter: lets a consumer tell "MY signal got a new
        # sample" from "the global version bumped because some OTHER logic node
        # published" -- e.g. a rolling monitor must append one point per sample
        # of its own source, not one per unrelated logic node tick.
        self._sig_version: dict[str, int] = {}

    # ------------------------------------------------------------- publish side
    def publish(self, values: Mapping[str, object], *, shot: int | None = None,
                provenance: int | None = None) -> int:
        """Record one shot's worth of named values; returns the new version.

        Every value is coerced to a float ndarray (scalars become 0-d) and COPIED,
        so the logic node may freely reuse its buffers.

        ``provenance`` is the SOURCE-shot id every value in this call belongs to (an acquiring node mints
        one via :meth:`next_source_shot`; a reactive processor passes the id of the input it consumed).
        ``None`` -> :data:`NO_LINEAGE` (a free-running scalar / health counter that joins no display shot).
        """

        with self._lock:
            self._shot = int(shot) if shot is not None else self._shot + 1
            src_id = int(provenance) if provenance is not None else NO_LINEAGE
            for name, value in values.items():
                key = str(name)
                arr = np.array(value, dtype=float, copy=True)
                ring = self._signals.get(key)
                if ring is None:
                    ring = deque(maxlen=self._history_len)
                    self._signals[key] = ring
                    self._src[key] = deque(maxlen=self._history_len)
                ring.append(arr)
                self._src[key].append(src_id)       # lockstep with ring: value[-1] was acquired at src[-1]
                self._sig_version[key] = self._sig_version.get(key, 0) + 1
            self._version += 1
            return self._version

    def next_source_shot(self) -> int:
        """Mint a fresh monotonic SOURCE-shot id (an acquiring node calls this once per real shot, then
        tags every signal of that shot with it; reactive processors inherit it instead of minting)."""
        with self._lock:
            self._source_shot += 1
            return self._source_shot

    def latest_provenance(self, name: str) -> int:
        """Source-shot id of ``name``'s most recent sample, or :data:`NO_LINEAGE` if absent/empty.
        SOFT-fails (never raises) so a reactive reader inheriting an id just no-ops on a missing input."""
        with self._lock:
            src = self._src.get(str(name))
            return int(src[-1]) if src else NO_LINEAGE

    def provenance_map(self) -> dict[str, int]:
        """``{name: latest source-shot id}`` for every signal -- the cheap id-only read the display clock
        uses to choose a coherent shot (no value copies)."""
        with self._lock:
            return {name: int(src[-1]) for name, src in self._src.items() if src}

    def clear(self) -> None:
        with self._lock:
            self._signals.clear()
            self._src.clear()
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
                    self._src.pop(name, None)          # lockstep: a leftover would mis-tag a future same-named signal
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
        """One consistent {name: latest value} mapping (the expression namespace).

        LATEST of each signal independently -- NOT coherent across producers (a derived ``frame_judged``
        lags the ``frame`` it came from).  Use :meth:`snapshot_at` for a shot-coherent display; this stays
        for the notebook expression namespace + tests that want the rawest values."""

        with self._lock:
            names = list(self._signals)
        out: dict[str, object] = {}
        for name in names:
            try:
                out[name] = self.latest(name)
            except KeyError:  # pragma: no cover - racy clear() between list and read
                continue
        return out

    def snapshot_at(self, target: int | None) -> dict[str, object]:
        """One ``{name: value}`` mapping COHERENT at source-shot ``target``: each signal's value whose
        provenance == ``target``, else (signal absent at that shot, or :data:`NO_LINEAGE`) its latest value.

        This is the shot-coherent display read: with ``target`` = the newest shot every bound lineage
        signal has reached, ``frame`` and its derived ``frame_judged``/``occupied`` all resolve to the SAME
        physical shot, so the 2-D image and the site map can never show different shots.  ``target`` None ->
        latest of each (== :meth:`snapshot_latest`).  ONE locked read so value/id can't race a publish."""

        with self._lock:
            names = list(self._signals)
            out: dict[str, object] = {}
            for name in names:
                ring = self._signals.get(name)
                if not ring:
                    continue
                value = ring[-1]                      # default: latest (NO_LINEAGE / not-present-at-target)
                src = self._src.get(name)
                # Only a LINEAGE signal can resolve to a specific shot.  A free-running NO_LINEAGE signal
                # (loading rate / a static aux) belongs to NO shot, so it always stays at its
                # latest -- skip the scan entirely.  (Its whole ring is NO_LINEAGE, so the early-break below
                # would never fire and the loop would walk the FULL ring EVERY tick; combined with deque
                # indexing -- O(n) per element -- that was an O(n^2) per-signal freeze of the GUI thread.)
                if target is not None and src and src[-1] != NO_LINEAGE:
                    # ids only grow over the ring; walk from the NEWEST via reversed() iterators (O(1) per
                    # step on a deque -- NOT src[i] / ring[i] indexing, which is O(n) per access) and stop
                    # once a real id drops below target (it cannot reappear earlier in the ring).
                    for v, s in zip(reversed(ring), reversed(src)):
                        if s == target:
                            value = v
                            break
                        if s != NO_LINEAGE and s < target:
                            break
                out[name] = float(value) if value.ndim == 0 else value.copy()
        return out


__all__ = ["SignalHub", "DEFAULT_HISTORY", "NO_LINEAGE"]
