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
from typing import Callable, Mapping

from zlc_data import OwnedSnapshot

__all__ = [
    "ConsoleDataFront",
    "ConsoleDataPlane",
    "ConsoleSignalValue",
    "single_output_projection",
]


SnapshotProjector = Callable[[OwnedSnapshot], Mapping[str, OwnedSnapshot]]


def single_output_projection(output_name: str) -> SnapshotProjector:
    """Return the explicit identity projection for a one-output live route."""

    name = str(output_name)
    if not name or name.strip() != name:
        raise ValueError("live output name must be canonical non-empty text")

    def project(snapshot: OwnedSnapshot) -> Mapping[str, OwnedSnapshot]:
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("live projection source must be OwnedSnapshot")
        return {name: snapshot}

    return project


@dataclass(frozen=True)
class ConsoleSignalValue:
    """One signal at one producer-owned immutable revision."""

    name: str
    source: str                     # the node title that produced it
    snapshot: object                # OwnedSnapshot -- the (ref, block) pair a render needs
    coverage: object | None         # MonitorCoverage, or None for a scalar-less signal
    # Lineage, carried because only the freeze knows it: a renderer stamps what
    # it drew with the run and event it came from, and a value that lost these
    # on the way to a panel could only be presented with an invented one.
    run_id: object
    epoch_id: object                # causation domain the run belongs to
    join_digest: str                # exact immutable source/coherence digest
    presentation: object | None = None

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


class ConsoleDataPlane:
    """Own live slots and coalesce their revision notifications.

    Slots arrive from the RUN seam: a node's start closure builds one and
    registers it here, so this plane never talks to the domain itself -- it
    holds what the monitor handed back and reads it atomically.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[int, tuple[object, object, SnapshotProjector]] = {}
        self._dirty: set[int] = set()
        self._cache: dict[int, dict[str, ConsoleSignalValue]] = {}
        self._finals: dict[
            int,
            tuple[object, dict[str, ConsoleSignalValue]],
        ] = {}
        self._processors: dict[
            int,
            tuple[object, dict[str, ConsoleSignalValue]],
        ] = {}
        self._panels: dict[str, dict[str, ConsoleSignalValue]] = {}
        self._failures: dict[int, str] = {}
        self._membership_changed = False
        self._closed = False
        empty = MappingProxyType({})
        self._front = ConsoleDataFront(signals=empty, failures=empty)

    # ------------------------------------------------------------ membership
    def attach(
        self,
        node,
        slot,
        *,
        project_snapshot: SnapshotProjector,
    ) -> None:
        if slot is None:
            raise ValueError("a monitor slot is required")
        if not callable(project_snapshot):
            raise TypeError("live project_snapshot must be callable")
        key = id(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("console data plane is closed")
            if key in self._slots:
                raise RuntimeError(
                    "console node already owns a run-scoped live route"
                )
            self._slots[key] = (node, slot, project_snapshot)
            # The view factory attaches before the domain binds its materializer.
            # Only the slot's first real revision marks it dirty; trying to freeze
            # here would turn the normal ARMED/no-frame-yet state into a false
            # "no active dataset" failure (notably for an externally triggered
            # main camera waiting for PulseGUI).
            self._dirty.discard(key)
            self._cache.pop(key, None)
            self._failures.pop(key, None)
            self._membership_changed = True

    def mark_changed(self, node) -> None:
        """Mark one producer dirty from its worker-safe change listener."""

        key = id(node)
        with self._lock:
            if key in self._slots:
                self._dirty.add(key)

    def publish_final(self, node, projected: Mapping[str, object]) -> None:
        """Admit one successful Run's already-materialized FINAL datasets.

        ``projected`` is keyed by the catalog's bare output names.  The data
        plane qualifies them with the exact producer instance just like a live
        slot; it never invents an output that the node did not declare.
        """

        from .result_projection import ProjectedFinalSignal

        if not isinstance(projected, Mapping):
            raise TypeError("projected FINAL signals must be a mapping")
        declarations = tuple(
            getattr(node, "output_declarations", ()) or ()
        )
        run_scoped = {
            str(output.name)
            for output in declarations
            if bool(getattr(output, "run_scoped", False))
        }
        # A task may declare both a provisional RUN value and several FINAL
        # artifacts.  Every other live producer publishes its complete frozen
        # vocabulary.  This is an equality contract: accepting only a subset
        # would make a Camera configured for three events appear to work while
        # silently omitting frame_1/frame_2 from the data plane.
        declared = run_scoped or {
            str(output.name) for output in declarations
        }
        unknown = set(map(str, projected)).difference(declared)
        if unknown:
            raise ValueError(
                "FINAL projection contains undeclared outputs: "
                + ", ".join(sorted(unknown))
            )
        title = str(getattr(node, "name", "") or type(node).__name__)
        handle = getattr(node, "handle", None)
        run_id_value = getattr(handle, "run_id", None)
        run_id = getattr(run_id_value, "value", run_id_value)
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(
                "a successful FINAL projection must retain its RunHandle RunId"
            )
        frozen: dict[str, ConsoleSignalValue] = {}
        for output_name, value in projected.items():
            if not isinstance(value, ProjectedFinalSignal):
                raise TypeError(
                    "FINAL projection values must be ProjectedFinalSignal"
                )
            key = node.signal_key(str(output_name))
            snapshot = value.snapshot
            frozen[key] = ConsoleSignalValue(
                name=key,
                source=title,
                snapshot=snapshot,
                coverage=None,
                run_id=run_id,
                epoch_id=snapshot.ref.stream_generation.value,
                join_digest=value.join_digest,
                presentation=value.presentation,
            )
        key = id(node)
        with self._lock:
            self._finals[key] = (node, frozen)
            self._membership_changed = True

    def publish_processor(
        self,
        node,
        values: Mapping[str, ConsoleSignalValue],
    ) -> None:
        """Atomically replace one reactive Processor's complete output pair.

        The Processor supplies already-qualified immutable values because it is
        the owner that knows the exact input lineage.  This data plane validates
        only catalog ownership and swaps all declared outputs together; it does
        not schedule, reacquire, or reinterpret the source dataset.
        """

        if not isinstance(values, Mapping):
            raise TypeError("processor values must be a mapping")
        declared = tuple(node.published_signals())
        if set(map(str, values)) != set(declared):
            raise ValueError(
                "reactive Processor must publish its complete declared output set"
            )
        frozen: dict[str, ConsoleSignalValue] = {}
        for name in declared:
            value = values[name]
            if not isinstance(value, ConsoleSignalValue):
                raise TypeError(
                    "processor values must contain ConsoleSignalValue"
                )
            if value.name != name:
                raise ValueError(
                    "processor output key differs from ConsoleSignalValue.name"
                )
            frozen[name] = value
        key = id(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("console data plane is closed")
            self._processors[key] = (node, frozen)
            self._membership_changed = True

    def publish_panel(
        self,
        panel_id: str,
        values: Mapping[str, ConsoleSignalValue],
    ) -> None:
        """Atomically replace one Figure panel's complete derived signal set."""

        from zlc_data.console_records import panel_signal_key

        identity = str(panel_id).strip()
        if not identity:
            raise ValueError("panel_id must not be empty")
        if not isinstance(values, Mapping):
            raise TypeError("panel values must be a mapping")
        if not values:
            raise ValueError("use withdraw_panel() to remove panel outputs")
        prefix = f"@panel/{identity}/"
        frozen: dict[str, ConsoleSignalValue] = {}
        for raw_name, value in values.items():
            name = str(raw_name)
            if not isinstance(value, ConsoleSignalValue):
                raise TypeError("panel values must contain ConsoleSignalValue")
            if name != value.name:
                raise ValueError("panel output key differs from ConsoleSignalValue.name")
            if not name.startswith(prefix):
                raise ValueError("panel output belongs to a different panel")
            output_name = name[len(prefix) :]
            if panel_signal_key(identity, output_name) != name:
                raise ValueError("panel output key is not canonical")
            frozen[name] = value
        with self._lock:
            if self._closed:
                raise RuntimeError("console data plane is closed")
            self._panels[identity] = frozen
            self._membership_changed = True

    def withdraw_panel(self, panel_id: str) -> None:
        """Remove every derived signal owned by one Figure panel."""

        identity = str(panel_id).strip()
        if not identity:
            raise ValueError("panel_id must not be empty")
        with self._lock:
            if self._panels.pop(identity, None) is not None:
                self._membership_changed = True

    def detach(self, node) -> None:
        key = id(node)
        with self._lock:
            entry = self._slots.pop(key, None)
            self._dirty.discard(key)
            self._cache.pop(key, None)
            self._finals.pop(key, None)
            self._processors.pop(key, None)
            self._failures.pop(key, None)
            self._membership_changed = True
        if entry is not None:
            _node, slot, _project_snapshot = entry
            slot.close()

    def detach_live(self, node) -> None:
        """Withdraw only one node's run-scoped live route, retaining FINAL data."""

        key = id(node)
        with self._lock:
            entry = self._slots.pop(key, None)
            self._dirty.discard(key)
            self._cache.pop(key, None)
            self._failures.pop(key, None)
            if entry is not None:
                self._membership_changed = True
        if entry is not None:
            _node, slot, _project_snapshot = entry
            slot.close()

    def close(self) -> None:
        """Release every live slot owned by this console."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = tuple(self._slots.values())
            self._slots.clear()
            self._dirty.clear()
            self._cache.clear()
            self._finals.clear()
            self._processors.clear()
            self._panels.clear()
            self._failures.clear()
            self._membership_changed = True
        for _node, slot, _project_snapshot in entries:
            slot.close()

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
            node, slot, project_snapshot = slots[key]
            title = str(getattr(node, "name", "") or type(node).__name__)
            try:
                result = self._freeze_one(
                    node,
                    slot,
                    title,
                    project_snapshot=project_snapshot,
                )
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                with self._lock:
                    if self._slots.get(key) == (node, slot, project_snapshot):
                        self._cache.pop(key, None)
                        self._failures[key] = failure
                continue
            frozen, alignment_failure = result
            with self._lock:
                if self._slots.get(key) == (node, slot, project_snapshot):
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
            finals = {
                key: (node, dict(value))
                for key, (node, value) in self._finals.items()
            }
            processors = {
                key: (node, dict(value))
                for key, (node, value) in self._processors.items()
            }
            panels = {
                panel_id: dict(value)
                for panel_id, value in self._panels.items()
            }
            failed = dict(self._failures)
        for key, (node, _slot, _project_snapshot) in current.items():
            signals.update(cached.get(key, {}))
            failure = failed.get(key)
            if failure is not None:
                title = str(getattr(node, "name", "") or type(node).__name__)
                failures[title] = failure
        for _key, (_node, values) in finals.items():
            signals.update(values)
        for _key, (_node, values) in processors.items():
            signals.update(values)
        for _panel_id, values in panels.items():
            signals.update(values)
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
        *,
        project_snapshot: SnapshotProjector,
    ) -> tuple[dict[str, ConsoleSignalValue], str | None]:
        """One slot's atomic transaction, projected onto its declared outputs.

        A camera monitor publishes only its raw dataset.  Figure-owned Area,
        locked Cross, and Fit branches are derived later from the exact accepted
        panel front and never reconfigure this producer.
        """

        from zlc_data import dataset_revision_ref_to_tree
        from zlc_neutral_atom.runtime.dataset import (
            DatasetPreviewSnapshot,
            MonitorDatasetSnapshot,
        )
        from zlc_storage import canonical_digest

        run_id, causation, snapshot = slot.freeze_current()
        if not isinstance(run_id, str) or not run_id:
            raise TypeError("live dataset run_id must be a non-empty string")
        if not isinstance(causation, str) or not causation:
            raise TypeError(
                "live dataset causation_domain_id must be a non-empty string"
            )
        if isinstance(snapshot, MonitorDatasetSnapshot):
            head = snapshot.head
            if head is None:
                raise RuntimeError("monitor dataset has no accepted event head")
            join_digest = head.payload_digest
        elif isinstance(snapshot, DatasetPreviewSnapshot):
            join_digest = canonical_digest(
                {
                    "owner": "zlc_workbench.console-exact-preview",
                    "run_id": run_id,
                    "causation_domain_id": causation,
                    "revision": dataset_revision_ref_to_tree(snapshot.ref),
                    "coverage": {
                        "written_cells": snapshot.coverage.written_cells,
                        "total_cells": snapshot.coverage.total_cells,
                    },
                }
            )
        else:
            raise TypeError(
                "console live slot must freeze a typed monitor or exact preview snapshot"
            )
        source_snapshot = snapshot.snapshot
        projected = project_snapshot(source_snapshot)
        if not isinstance(projected, Mapping) or not projected:
            raise ValueError("live projection must return a non-empty mapping")
        declared = {
            str(output.name)
            for output in tuple(getattr(node, "output_declarations", ()) or ())
        }
        bare_names = tuple(projected)
        if any(
            not isinstance(name, str) or not name or name.strip() != name
            for name in bare_names
        ):
            raise ValueError("live projection output names must be canonical text")
        actual = set(bare_names)
        if actual != declared:
            raise ValueError(
                "live projection output set differs from its frozen declaration: "
                f"expected={tuple(sorted(declared))}, actual={tuple(sorted(actual))}"
            )
        frozen: dict[str, ConsoleSignalValue] = {}
        for output_name in bare_names:
            output_snapshot = projected[output_name]
            if not isinstance(output_snapshot, OwnedSnapshot):
                raise TypeError("live projection values must be OwnedSnapshot")
            if output_snapshot.ref.revision != source_snapshot.ref.revision:
                raise ValueError("live projection changed its source revision")
            if (
                output_snapshot.ref.stream_generation
                != source_snapshot.ref.stream_generation
            ):
                raise ValueError("live projection changed its source generation")
            selected = node.signal_key(output_name)
            frozen[selected] = ConsoleSignalValue(
                name=selected,
                source=title,
                snapshot=output_snapshot,
                coverage=snapshot.coverage,
                run_id=run_id,
                epoch_id=causation,
                join_digest=join_digest,
            )
        return frozen, getattr(slot, "notification_failure", None)
