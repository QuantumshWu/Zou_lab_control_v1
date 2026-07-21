"""MONITOR seam: the console's per-tick view of what its runs are producing.

Seam 3 of the composition root's rewiring contract (``app.py``).  A run node
publishes into a ``LiveDatasetSlot``; this module turns those slots into ONE
frozen snapshot per tick, which every panel, picker and legend then reads.

Freeze-latest, not a bus.  The old console read a mutable signal hub whenever it
felt like it, so two widgets in the same tick could disagree about which shot
they were showing.  Here the tick freezes once -- each slot materialises its own
atomic transaction -- and the resulting :class:`ConsoleTickSnapshot` is
immutable.  Two readers of one snapshot cannot disagree, and a slow consumer
holds a consistent past instead of a torn present.

What replaced the shot clock: a monitor tap overwrites when the display falls
behind rather than back-pressuring acquisition, and it says so per signal
through :class:`~zlc_neutral_atom.runtime.dataset.MonitorCoverage`
(written/total cells, missed events, current gap).  There is no global shot
counter to compare against, and reintroducing one would be a fiction -- signals
from different runs advance independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = ["ConsoleDataPlane", "ConsoleSignalValue", "ConsoleTickSnapshot"]


@dataclass(frozen=True)
class ConsoleSignalValue:
    """One signal as of one tick: its block, how current it is, and its version."""

    name: str
    source: str                     # the node title that produced it
    snapshot: object                # OwnedSnapshot -- the (ref, block) pair a render needs
    version: int                    # the producing stream's sequence at freeze time
    coverage: object | None         # MonitorCoverage, or None for a scalar-less signal
    coherent: bool                  # derived alongside the same raw event
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
class ConsoleTickSnapshot:
    """Everything the board may read this tick.  Immutable by construction."""

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
    """Owns the live slots and freezes them together, once per tick.

    Slots arrive from the RUN seam: a node's start closure builds one and
    registers it here, so this plane never talks to the domain itself -- it
    holds what the monitor handed back and reads it atomically.
    """

    def __init__(self) -> None:
        self._slots: dict[int, tuple[object, object]] = {}   # id(node) -> (node, slot)

    # ------------------------------------------------------------ membership
    def attach(self, node, slot) -> None:
        if slot is None:
            raise ValueError("a monitor slot is required")
        self._slots[id(node)] = (node, slot)

    def detach(self, node) -> None:
        self._slots.pop(id(node), None)

    def __len__(self) -> int:
        return len(self._slots)

    # ---------------------------------------------------------------- freeze
    def freeze(self) -> ConsoleTickSnapshot:
        """Materialise every attached slot into one immutable tick snapshot.

        A slot that cannot be frozen (its run went terminal, its dataset was
        withdrawn) contributes a FAILURE entry, never an exception: one dead
        source must not blank the whole board, and the operator still needs to
        see why that one row stopped moving.
        """

        signals: dict[str, ConsoleSignalValue] = {}
        failures: dict[str, str] = {}
        for node, slot in list(self._slots.values()):
            title = str(getattr(node, "name", "") or type(node).__name__)
            try:
                frozen = self._freeze_one(node, slot, title)
            except Exception as error:
                failures[title] = f"{type(error).__name__}: {error}"
                continue
            signals.update(frozen)
        return ConsoleTickSnapshot(signals=signals, failures=failures)

    def _freeze_one(self, node, slot, title: str) -> dict[str, ConsoleSignalValue]:
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
        coherent = (
            metadata is not None
            and raw is not None
            and getattr(metadata, "source_event_ref", None) == raw.head
        )
        out: dict[str, ConsoleSignalValue] = {}
        if raw is not None and declared:
            out[declared[0]] = ConsoleSignalValue(
                name=declared[0], source=title, snapshot=raw.snapshot,
                version=self._sequence(raw), coverage=raw.coverage, coherent=True,
                run_id=run_id, epoch_id=causation,
                join_digest=str(getattr(head, "payload_digest", "") or ""),
            )
        if scalar is not None and len(declared) > 1:
            out[declared[1]] = ConsoleSignalValue(
                name=declared[1], source=title, snapshot=scalar.snapshot,
                version=self._sequence(scalar), coverage=scalar.coverage,
                coherent=bool(coherent),
                run_id=run_id, epoch_id=causation,
                # A derived scalar names the raw event it reduced, so its join
                # digest is that event's -- not a second digest of its own.
                join_digest=str(getattr(head, "payload_digest", "") or ""),
            )
        return out

    @staticmethod
    def _sequence(dataset_snapshot) -> int:
        head = getattr(dataset_snapshot, "head", None)
        return int(getattr(head, "sequence", 0) or 0)
