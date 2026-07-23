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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from zlc_data import ReductionMethod, Selection
    from zlc_neutral_atom.monitor_application import CameraMonitorRoiState
    from zlc_neutral_atom.runtime.control import ControlReceipt

__all__ = ["ConsoleDataFront", "ConsoleDataPlane", "ConsoleSignalValue"]


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
    join_digest: str                # payload digest of the event this froze
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
        self._slots: dict[int, tuple[object, object]] = {}   # id(node) -> (node, slot)
        self._exact_watchers: dict[int, object] = {}
        self._exact_candidates: dict[int, tuple[str, str, object]] = {}
        self._exact_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="console-exact-preview",
        )
        self._dirty: set[int] = set()
        self._cache: dict[int, dict[str, ConsoleSignalValue]] = {}
        self._finals: dict[
            int,
            tuple[object, dict[str, ConsoleSignalValue]],
        ] = {}
        self._failures: dict[int, str] = {}
        self._membership_changed = False
        self._closed = False
        empty = MappingProxyType({})
        self._front = ConsoleDataFront(signals=empty, failures=empty)

    # ------------------------------------------------------------ membership
    def attach(self, node, slot) -> None:
        if slot is None:
            raise ValueError("a monitor slot is required")
        key = id(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("console data plane is closed")
            self._slots[key] = (node, slot)
            # The view factory attaches before the domain binds its materializer.
            # Only the slot's first real revision marks it dirty; trying to freeze
            # here would turn the normal ARMED/no-frame-yet state into a false
            # "no active dataset" failure (notably for an externally triggered
            # main camera waiting for PulseGUI).
            self._dirty.discard(key)
            self._exact_candidates.pop(key, None)
            self._cache.pop(key, None)
            self._failures.pop(key, None)
            self._membership_changed = True

    def attach_exact(self, node, slot) -> None:
        """Attach one exact provisional dataset without blocking the Qt owner."""

        from zlc_workbench.progressive_scan import ExactDatasetLiveSlot

        if not isinstance(slot, ExactDatasetLiveSlot):
            raise TypeError("exact console preview requires ExactDatasetLiveSlot")
        self.attach(node, slot)
        try:
            slot.set_change_listener(
                lambda: self._ensure_exact_watcher(node, slot)
            )
        except BaseException:
            self.detach(node)
            raise

    def _ensure_exact_watcher(self, node, slot) -> None:
        key = id(node)
        with self._lock:
            if self._closed or self._slots.get(key) != (node, slot):
                return
            if self._exact_watchers.get(key) is slot:
                return
            self._exact_watchers[key] = slot
            try:
                self._exact_executor.submit(self._watch_exact, node, slot)
            except BaseException:
                self._exact_watchers.pop(key, None)
                raise

    def _watch_exact(self, node, slot) -> None:
        """Wait for exact builder revisions on the data plane's sole view lane."""

        from zlc_data import DatasetRevision

        key = id(node)
        after = DatasetRevision(0)
        try:
            while True:
                candidate = slot.wait_and_freeze(after, timeout=0.1)
                if candidate is not None:
                    run_id, causation, snapshot = candidate
                    after = snapshot.ref.revision
                    with self._lock:
                        if self._slots.get(key) != (node, slot):
                            return
                        self._exact_candidates[key] = (
                            run_id,
                            causation,
                            snapshot,
                        )
                        self._dirty.add(key)
                if slot.terminal:
                    failure = slot.failure
                    if failure is not None:
                        with self._lock:
                            if self._slots.get(key) == (node, slot):
                                self._failures[key] = failure
                                self._dirty.add(key)
                    return
        except Exception as error:
            with self._lock:
                if self._slots.get(key) == (node, slot):
                    self._failures[key] = f"{type(error).__name__}: {error}"
                    self._dirty.add(key)
        finally:
            with self._lock:
                if self._exact_watchers.get(key) is slot:
                    self._exact_watchers.pop(key, None)

    def mark_changed(self, node) -> None:
        """Mark one producer dirty from its worker-safe change listener."""

        key = id(node)
        with self._lock:
            if key in self._slots:
                self._dirty.add(key)

    def submit_camera_roi_control(
        self,
        node: object,
        selection: Selection | None,
        reduction: ReductionMethod,
    ) -> ControlReceipt:
        """Submit one selector ROI to the exact attached camera-monitor slot.

        Membership is sampled under the data-plane lock, while preparation and
        publication run outside it.  A detach/replacement during either domain
        call invalidates the operation instead of returning a receipt for a
        producer this console no longer owns.
        """

        from zlc_data import ReductionMethod, Selection
        from zlc_neutral_atom.monitor_application import CameraMonitorViewSpec
        from zlc_neutral_atom.runtime.control import ControlReceipt
        from zlc_workbench.live import LiveDatasetSlot

        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        if not isinstance(reduction, ReductionMethod):
            raise TypeError("reduction must be ReductionMethod")
        key = id(node)
        with self._lock:
            entry = self._slots.get(key)
        if entry is None or entry[0] is not node:
            raise LookupError("camera monitor node has no attached live slot")
        slot = entry[1]
        if not isinstance(slot, LiveDatasetSlot) or not isinstance(
            getattr(slot, "spec", None),
            CameraMonitorViewSpec,
        ):
            raise TypeError("node is not attached to a camera monitor live slot")

        candidate = slot.prepare_camera_roi_control(selection, reduction)
        with self._lock:
            if self._slots.get(key) != (node, slot):
                raise RuntimeError(
                    "camera monitor membership changed while preparing ROI control"
                )
        receipt = slot.submit_camera_roi_control(candidate)
        if not isinstance(receipt, ControlReceipt):
            raise TypeError("camera monitor returned an invalid ControlReceipt")
        with self._lock:
            if self._slots.get(key) != (node, slot):
                raise RuntimeError(
                    "camera monitor membership changed while submitting ROI control"
                )
        return receipt

    def current_camera_roi_state(
        self,
        node: object,
    ) -> CameraMonitorRoiState:
        """Return the applied ROI branch for one exact attached monitor node."""

        from zlc_neutral_atom.monitor_application import (
            CameraMonitorRoiState,
            CameraMonitorViewSpec,
        )
        from zlc_workbench.live import LiveDatasetSlot

        key = id(node)
        with self._lock:
            entry = self._slots.get(key)
        if entry is None or entry[0] is not node:
            raise LookupError("camera monitor node has no attached live slot")
        slot = entry[1]
        if not isinstance(slot, LiveDatasetSlot) or not isinstance(
            getattr(slot, "spec", None),
            CameraMonitorViewSpec,
        ):
            raise TypeError("node is not attached to a camera monitor live slot")
        state = slot.current_camera_roi_state()
        if not isinstance(state, CameraMonitorRoiState):
            raise TypeError("camera monitor returned an invalid ROI state")
        with self._lock:
            if self._slots.get(key) != (node, slot):
                raise RuntimeError(
                    "camera monitor membership changed while reading ROI state"
                )
        return state

    def publish_final(self, node, projected: Mapping[str, object]) -> None:
        """Admit one successful Run's already-materialized FINAL datasets.

        ``projected`` is keyed by the catalog's bare output names.  The data
        plane qualifies them with the exact producer instance just like a live
        slot; it never invents an output that the node did not declare.
        """

        from .result_projection import ProjectedFinalSignal

        if not isinstance(projected, Mapping):
            raise TypeError("projected FINAL signals must be a mapping")
        declared = {
            str(output.name)
            for output in tuple(
                getattr(getattr(node, "spec", None), "declared_outputs", ()) or ()
            )
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

    def detach(self, node) -> None:
        key = id(node)
        with self._lock:
            entry = self._slots.pop(key, None)
            self._exact_watchers.pop(key, None)
            self._exact_candidates.pop(key, None)
            self._dirty.discard(key)
            self._cache.pop(key, None)
            self._finals.pop(key, None)
            self._failures.pop(key, None)
            self._membership_changed = True
        if entry is not None:
            _node, slot = entry
            slot.close()

    def close(self) -> None:
        """Release every live slot owned by this console."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = tuple(self._slots.values())
            self._slots.clear()
            self._exact_watchers.clear()
            self._exact_candidates.clear()
            self._dirty.clear()
            self._cache.clear()
            self._finals.clear()
            self._failures.clear()
            self._membership_changed = True
        for _node, slot in entries:
            slot.close()
        self._exact_executor.shutdown(wait=True, cancel_futures=True)

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
            finals = {
                key: (node, dict(value))
                for key, (node, value) in self._finals.items()
            }
            failed = dict(self._failures)
        for key, (node, _slot) in current.items():
            signals.update(cached.get(key, {}))
            failure = failed.get(key)
            if failure is not None:
                title = str(getattr(node, "name", "") or type(node).__name__)
                failures[title] = failure
        for _key, (_node, values) in finals.items():
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
    ) -> tuple[dict[str, ConsoleSignalValue], str | None]:
        """One slot's atomic transaction, projected onto its declared outputs.

        A camera monitor freezes RAW plus an optional derived ROI SCALAR in one
        transaction; the node's catalog spec declares those two outputs in that
        order, so the projection is positional against the declaration rather
        than against names invented here.
        """

        from zlc_neutral_atom.monitor_application import CameraMonitorViewSpec
        from zlc_neutral_atom.runtime.pipeline import (
            CapturePreviewSpec,
            ExactDatasetPreviewSpec,
        )

        spec = getattr(slot, "spec", None)
        if isinstance(spec, CapturePreviewSpec):
            return self._freeze_capture_preview(node, slot, title)
        if isinstance(spec, ExactDatasetPreviewSpec):
            return self._freeze_exact_preview(node, slot, title)
        if not isinstance(spec, CameraMonitorViewSpec):
            raise TypeError(
                "console live slot must own a camera monitor, capture preview, "
                "or exact dataset preview"
            )
        run_id, causation, snapshot = slot.freeze_camera_current()
        declared = tuple(node.published_signals())
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
                coverage=raw.coverage,
                run_id=run_id, epoch_id=causation,
                join_digest=str(getattr(head, "payload_digest", "") or ""),
            )
        alignment_failure = getattr(slot, "notification_failure", None)
        if scalar is not None and len(declared) > 1 and scalar_matches_raw:
            out[declared[1]] = ConsoleSignalValue(
                name=declared[1], source=title, snapshot=scalar.snapshot,
                coverage=scalar.coverage,
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
            scalar_failure = (
                f"{declared[1]} does not identify the raw event it reduced"
            )
            alignment_failure = (
                scalar_failure
                if alignment_failure is None
                else f"{alignment_failure}; {scalar_failure}"
            )
        return out, alignment_failure

    def _freeze_capture_preview(
        self,
        node,
        slot,
        title: str,
    ) -> tuple[dict[str, ConsoleSignalValue], str | None]:
        """Freeze the exact capture preview without pretending it has ROI output."""

        run_id, causation, snapshot = slot.freeze_current()
        declared = tuple(node.published_signals())
        if len(declared) != 1:
            raise ValueError("camera capture must declare exactly one preview output")
        head = snapshot.head
        return {
            declared[0]: ConsoleSignalValue(
                name=declared[0],
                source=title,
                snapshot=snapshot.snapshot,
                coverage=snapshot.coverage,
                run_id=run_id,
                epoch_id=causation,
                join_digest=str(
                    getattr(head, "payload_digest", "") or ""
                ),
            )
        }, getattr(slot, "notification_failure", None)

    def _freeze_exact_preview(
        self,
        node,
        slot,
        title: str,
    ) -> tuple[dict[str, ConsoleSignalValue], str | None]:
        """Project the occupancy builder's exact provisional counts dataset."""

        from zlc_data import dataset_revision_ref_to_tree
        from zlc_storage import canonical_digest

        key = id(node)
        with self._lock:
            candidate = self._exact_candidates.get(key)
        if candidate is None:
            raise RuntimeError("exact preview has no materialized revision")
        run_id, causation, snapshot = candidate
        outputs = tuple(
            getattr(getattr(node, "spec", None), "declared_outputs", ()) or ()
        )
        published = tuple(node.published_signals())
        counts = [
            full
            for full, output in zip(published, outputs, strict=True)
            if str(output.name) == "counts"
        ]
        if len(counts) != 1:
            raise ValueError(
                "exact occupancy preview requires one declared counts output"
            )
        name = counts[0]
        return {
            name: ConsoleSignalValue(
                name=name,
                source=title,
                snapshot=snapshot.snapshot,
                coverage=snapshot.coverage,
                run_id=run_id,
                epoch_id=causation,
                join_digest=canonical_digest(
                    dataset_revision_ref_to_tree(snapshot.ref)
                ),
            )
        }, slot.failure
