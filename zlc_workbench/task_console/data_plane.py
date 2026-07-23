"""MONITOR seam: change-driven latest fronts from independent producers.

Seam 3 of the composition root's rewiring contract (``app.py``).  A run node
publishes into a ``LiveDatasetSlot``; this module freezes only slots that
reported a new revision.  Each slot is one producer transaction.  Combining
their latest immutable fronts into one presentation cycle does *not* assert
that independent producers observed the same physical shot.

Freeze-latest, not a bus.  The old console read a mutable signal hub whenever it
felt like it, so two widgets could disagree about which revision of one signal
they were showing.  Each changed slot materialises its own atomic transaction
exactly once; unchanged slots reuse their immutable fronts.  The resulting
:class:`ConsoleDataFront` is immutable, so readers agree about every individual
producer revision without inventing cross-producer causation.

What replaced the shot clock: a monitor tap overwrites when the display falls
behind rather than back-pressuring acquisition, and it says so per signal
through :class:`~zlc_neutral_atom.runtime.dataset.MonitorCoverage`
(written/total cells, missed events, current gap).  There is no global shot
counter to compare against, and reintroducing one would be a fiction -- signals
from different runs advance independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Mapping

__all__ = ["ConsoleDataFront", "ConsoleDataPlane", "ConsoleSignalValue"]


@dataclass(frozen=True)
class ConsoleSignalValue:
    """One signal at one producer-owned immutable revision."""

    name: str
    source: str                     # the node title that produced it
    snapshot: object                # OwnedSnapshot -- the (ref, block) pair a render needs
    version: int                    # the producing stream's sequence at freeze time
    coverage: object | None         # MonitorCoverage, or None for a scalar-less signal
    # Lineage, carried because only the freeze knows it: a renderer stamps what
    # it drew with the run and event it came from, and a value that lost these
    # on the way to a panel could only be presented with an invented one.
    run_id: object
    epoch_id: object                # causation domain the run belongs to
    join_digest: str                # payload digest of the event this froze

    # The block is the value; these read off it rather than copying, so a panel
    # and a legend describing "the same signal" cannot describe different data.
    @property
    def block(self):
        """The snapshot's DataBlock -- shape/dtype/schema live here."""

        return getattr(self.snapshot, "block", None)

    @property
    def schema(self):
        return getattr(self.block, "schema", None)

    @property
    def values(self):
        """The block's array.  Read-only by ownership: never mutate a frozen block."""

        return getattr(self.block, "values", None)

    @property
    def shape(self) -> tuple[int, ...]:
        values = self.values
        return tuple(getattr(values, "shape", ()) or ())

    @property
    def cell_schema(self):
        """The per-cell value schema -- where dtype / unit / data axes live.

        A DatasetSchema describes the DATASET (repeat axis, point axes, layout);
        what a cell actually holds is its ``cell_schema``.  Reading through it
        keeps the console on the same description the producer declared.
        """

        return getattr(self.schema, "cell_schema", None)

    @property
    def dtype(self):
        return getattr(self.cell_schema, "dtype", None)

    @property
    def unit(self) -> str | None:
        return getattr(self.cell_schema, "value_unit", None)

    @property
    def axes(self) -> tuple:
        return tuple(getattr(self.cell_schema, "data_axes", ()) or ())

    @property
    def behind(self) -> int:
        """How many events the tap dropped for this signal -- 0 when keeping up.

        This is what the display-behind advisory reads.  It is per signal, from
        the tap that actually dropped them; there is no global shot counter to
        subtract, and inventing one would mean comparing runs that advance
        independently.
        """

        return int(getattr(self.coverage, "missed_events", 0) or 0)


@dataclass(frozen=True)
class ConsoleDataFront:
    """Latest immutable value of each producer; no cross-producer join claim."""

    signals: Mapping[str, ConsoleSignalValue]
    failures: Mapping[str, str]     # node title -> why its freeze did not happen

    def names(self) -> tuple[str, ...]:
        return tuple(self.signals)

    def value(self, name: str) -> ConsoleSignalValue | None:
        return self.signals.get(str(name))

    def versions(self) -> dict[str, int]:
        """Per-signal version, for change detection.

        Derived from the producing stream's sequence rather than counted here:
        a version that the console incremented itself would advance even when
        the tap dropped the event, which is exactly the case a panel needs to
        distinguish.
        """

        return {name: value.version for name, value in self.signals.items()}


class ConsoleDataPlane:
    """Own live slots and coalesce their revision notifications.

    Slots arrive from the RUN seam: a node's start closure builds one and
    registers it here, so this plane never talks to the domain itself -- it
    holds what the monitor handed back and reads it atomically.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[int, tuple[object, object]] = {}   # id(node) -> (node, slot)
        self._dirty: set[int] = set()
        self._cache: dict[int, dict[str, ConsoleSignalValue]] = {}
        self._failures: dict[int, str] = {}
        self._membership_changed = False
        empty = MappingProxyType({})
        self._front = ConsoleDataFront(signals=empty, failures=empty)

    # ------------------------------------------------------------ membership
    def attach(self, node, slot) -> None:
        if slot is None:
            raise ValueError("a monitor slot is required")
        key = id(node)
        with self._lock:
            self._slots[key] = (node, slot)
            self._dirty.add(key)
            self._cache.pop(key, None)
            self._failures.pop(key, None)
            self._membership_changed = True

    def mark_changed(self, node) -> None:
        """Mark one producer dirty from its worker-safe change listener."""

        key = id(node)
        with self._lock:
            if key in self._slots:
                self._dirty.add(key)

    def detach(self, node) -> None:
        key = id(node)
        with self._lock:
            self._slots.pop(key, None)
            self._dirty.discard(key)
            self._cache.pop(key, None)
            self._failures.pop(key, None)
            self._membership_changed = True

    def __len__(self) -> int:
        with self._lock:
            return len(self._slots)

    # ---------------------------------------------------------------- freeze
    def freeze(self) -> ConsoleDataFront:
        """Return the current immutable board front, advancing changed sources.

        A slot that cannot be frozen (its run went terminal, its dataset was
        withdrawn) contributes a FAILURE entry, never an exception: one dead
        source must not blank the whole board, and the operator still needs to
        see why that one row stopped moving.

        With no producer revision or membership change this returns the exact
        same object.  The GUI timer is therefore only a polling clock; it cannot
        manufacture a new aggregate data front merely because time passed.
        """

        with self._lock:
            if not self._dirty and not self._membership_changed:
                return self._front
            slots = dict(self._slots)
            dirty = self._dirty.intersection(slots)
            self._dirty.difference_update(dirty)
            # Consume only the membership state observed above.  A concurrent
            # attach/detach after this lock is released sets it again and is
            # therefore rebuilt on the next owner tick.
            self._membership_changed = False
        for key in dirty:
            node, slot = slots[key]
            title = str(getattr(node, "name", "") or type(node).__name__)
            try:
                frozen, alignment_failure = self._freeze_one(node, slot, title)
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                with self._lock:
                    if self._slots.get(key) == (node, slot):
                        self._cache.pop(key, None)
                        self._failures[key] = failure
                continue
            with self._lock:
                if self._slots.get(key) == (node, slot):
                    self._cache[key] = frozen
                    if alignment_failure is None:
                        self._failures.pop(key, None)
                    else:
                        self._failures[key] = alignment_failure
        signals: dict[str, ConsoleSignalValue] = {}
        failures: dict[str, str] = {}
        with self._lock:
            current = dict(self._slots)
            cached = {key: dict(value) for key, value in self._cache.items()}
            failed = dict(self._failures)
        for key, (node, _slot) in current.items():
            signals.update(cached.get(key, {}))
            failure = failed.get(key)
            if failure is not None:
                title = str(getattr(node, "name", "") or type(node).__name__)
                failures[title] = failure
        front = ConsoleDataFront(
            signals=MappingProxyType(signals),
            failures=MappingProxyType(failures),
        )
        with self._lock:
            self._front = front
        return front

    def _freeze_one(
        self,
        node,
        slot,
        title: str,
    ) -> tuple[dict[str, ConsoleSignalValue], str | None]:
        """One slot's atomic transaction, projected onto its declared outputs.

        A camera monitor freezes RAW plus an optional derived ROI SCALAR in one
        transaction; the node's catalog spec declares those two outputs in that
        order, so the projection is positional against the declaration rather
        than against names invented here.
        """

        run_id, causation, snapshot = slot.freeze_camera_current()
        declared = tuple(
            str(decl.name)
            for decl in getattr(getattr(node, "spec", None), "declared_outputs", ()) or ()
        )
        raw = snapshot.raw
        head = None if raw is None else raw.head
        scalar = snapshot.scalar
        metadata = snapshot.scalar_metadata
        # Same-transaction check, the presentation-layer half of coherence: the
        # scalar is only "of this frame" when it names the raw event it reduced.
        scalar_matches_raw = (
            metadata is not None
            and raw is not None
            and getattr(metadata, "source_event_ref", None) == raw.head
        )
        out: dict[str, ConsoleSignalValue] = {}
        if raw is not None and declared:
            out[declared[0]] = ConsoleSignalValue(
                name=declared[0], source=title, snapshot=raw.snapshot,
                version=self._sequence(raw), coverage=raw.coverage,
                run_id=run_id, epoch_id=causation,
                join_digest=str(getattr(head, "payload_digest", "") or ""),
            )
        alignment_failure = None
        if scalar is not None and len(declared) > 1 and scalar_matches_raw:
            out[declared[1]] = ConsoleSignalValue(
                name=declared[1], source=title, snapshot=scalar.snapshot,
                version=self._sequence(scalar), coverage=scalar.coverage,
                run_id=run_id, epoch_id=causation,
                # A derived scalar names the raw event it reduced, so its join
                # digest is that event's -- not a second digest of its own.
                join_digest=str(getattr(head, "payload_digest", "") or ""),
            )
        elif scalar is not None and len(declared) > 1:
            # Never stamp an unrelated derived value with the raw event's join
            # digest.  Keep the valid raw front and expose the missing derived
            # branch as a producer failure instead of drawing a plausible but
            # falsely aligned scalar.
            alignment_failure = (
                f"{declared[1]} does not identify the raw event it reduced"
            )
        return out, alignment_failure

    @staticmethod
    def _sequence(dataset_snapshot) -> int:
        head = getattr(dataset_snapshot, "head", None)
        return int(getattr(head, "sequence", 0) or 0)
