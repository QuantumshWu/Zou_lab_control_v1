"""Typed named-signal transport between experiment logic and consumers.

Every stored signal is a :class:`~.signal_tensor.SignalTensor` whose physical
array is exactly ``(R, P, *data_shape)``.  A producer registers a
:class:`~.signal_tensor.SignalSchema`; the hub is the single normalization and
validation boundary.  Raw unregistered publishes remain useful for notebooks,
but have one fixed meaning (R=1, P=1, the whole value is one datum) rather than
an ndarray-rank guess.

The hub stores one current tensor per name.  Cumulative acquisitions update it
with :class:`~.signal_tensor.TensorPatch`, and a forward delta journal replays
history and stateful subscriptions lazily.  Consequently a P-point scan consumes O(P*D)
payload instead of retaining P copies of its O(P*D) cumulative block.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import heapq
import threading
import time

import numpy as np

from .signal_tensor import (
    SignalCursor,
    SignalHistoryGap,
    SignalSchema,
    SignalTensor,
    TensorHistory,
    TensorPatch,
    TensorStore,
)


DEFAULT_HISTORY = 2048

# The lineage sentinel is shared vocabulary, not a signal-hub concept: the
# render layer reads it too.  Defined once in zlc_data, imported here.
from zlc_data.vocabulary import NO_LINEAGE


@dataclass(frozen=True)
class CoherentUpdate:
    """The earliest unconsumed lineage-coherent signal update.

    ``tensors`` are exact historical states at the selected per-signal cursor,
    not aliases of latest values.  ``cursors`` are stateful subscription handles
    passed to the next call.  For one signal, provenance may be ``NO_LINEAGE``;
    multiple signals are returned only when they share one real provenance id.
    """

    provenance: int
    tensors: Mapping[str, SignalTensor]
    cursors: Mapping[str, "SignalSubscriptionCursor"]

    def values(self, *, logical: bool = False) -> dict[str, np.ndarray]:
        return {
            name: (tensor.logical() if logical else tensor.data.copy())
            for name, tensor in self.tensors.items()
        }


class _SubscribedSignal:
    """Mutable forward-replay state for one name inside a subscription."""

    def __init__(
        self,
        name: str,
        cursor: SignalCursor,
        data: np.ndarray | None,
        valid: np.ndarray | None,
    ):
        self.name = str(name)
        self.schema_version = int(cursor.schema_version)
        self.version = int(cursor.version)
        self.scan_version = int(cursor.version)
        self.data = data
        self.valid = valid
        self.pending: deque = deque()
        self.provenance_versions: dict[int, deque[int]] = {}

    @property
    def cursor(self) -> SignalCursor:
        return SignalCursor(self.schema_version, self.version)

    @property
    def scan_cursor(self) -> SignalCursor:
        return SignalCursor(self.schema_version, self.scan_version)

    def append(self, ref) -> bool:
        expected = self.scan_version + 1
        if ref.cursor.version != expected:
            raise SignalHistoryGap(
                f"subscription for {self.name!r} expected update {expected}, "
                f"got {ref.cursor.version}.")
        self.pending.append(ref)
        self.scan_version = int(ref.cursor.version)
        provenance = int(ref.provenance)
        if provenance == NO_LINEAGE:
            return False
        versions = self.provenance_versions.setdefault(provenance, deque())
        was_absent = not versions
        versions.append(int(ref.cursor.version))
        return was_absent

    def first_version(self, provenance: int) -> int | None:
        versions = self.provenance_versions.get(int(provenance))
        return None if not versions else int(versions[0])

    def remove_pending_ref(self, ref) -> None:
        provenance = int(ref.provenance)
        if provenance == NO_LINEAGE:
            return
        versions = self.provenance_versions.get(provenance)
        if not versions or int(versions[0]) != int(ref.cursor.version):
            raise SignalHistoryGap(
                f"subscription provenance index for {self.name!r} is inconsistent at "
                f"update {ref.cursor.version}.")
        versions.popleft()
        if not versions:
            self.provenance_versions.pop(provenance, None)


class SignalSubscriptionCursor:
    """One signal's position handle inside a stateful subscription."""

    def __init__(self, subscription: "SignalSubscription", name: str):
        self._subscription = subscription
        self.name = str(name)

    @property
    def schema_version(self) -> int:
        return self._subscription._states[self.name].schema_version

    @property
    def version(self) -> int:
        return self._subscription._states[self.name].version

    @property
    def position(self) -> SignalCursor:
        return SignalCursor(self.schema_version, self.version)

    def __repr__(self) -> str:
        return (
            f"SignalSubscriptionCursor(name={self.name!r}, "
            f"schema_version={self.schema_version}, version={self.version})"
        )


class SignalSubscription(Mapping[str, SignalSubscriptionCursor]):
    """Stateful coherent reader over one fixed ordered set of signal names.

    The session owns one reconstructed mutable tensor per signal plus compact
    pending update metadata.  Each retained delta is discovered and applied at
    most once.  Mapping values are lightweight handles back to this shared
    session, so ``dict(update.cursors)`` preserves state for existing callers.
    """

    def __init__(self, hub: "SignalHub", names: tuple[str, ...], states):
        self._hub = hub
        self.names = tuple(names)
        self._states: dict[str, _SubscribedSignal] = dict(states)
        self._handles = {
            name: SignalSubscriptionCursor(self, name)
            for name in self.names
        }
        self._common: set[int] = set()
        self._common_heap: list[tuple[int, int]] = []

    def __getitem__(self, name: str) -> SignalSubscriptionCursor:
        return self._handles[str(name)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def _sync_common(self, provenance: int, *, refresh_position: bool = False) -> None:
        provenance = int(provenance)
        if provenance == NO_LINEAGE:
            return
        common = all(
            state.first_version(provenance) is not None
            for state in self._states.values()
        )
        if not common:
            self._common.discard(provenance)
            return
        first_version = self._states[self.names[0]].first_version(provenance)
        assert first_version is not None
        if provenance not in self._common or refresh_position:
            self._common.add(provenance)
            heapq.heappush(self._common_heap, (first_version, provenance))

    def _next_common(self) -> int | None:
        first = self._states[self.names[0]]
        while self._common_heap:
            version, provenance = self._common_heap[0]
            current = first.first_version(provenance)
            if provenance not in self._common or current != version:
                heapq.heappop(self._common_heap)
                continue
            return int(provenance)
        return None


class HistoryReservation:
    """Scoped consumer claim on minimum per-signal journal capacity."""

    def __init__(self, hub: "SignalHub", token: int, names: tuple[str, ...], capacity: int):
        self._hub = hub
        self._token = int(token)
        self.names = names
        self.capacity = int(capacity)
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._hub._release_history_reservation(self._token, self.names)

    def __enter__(self) -> "HistoryReservation":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class SignalHub:
    """Thread-safe schema registry and typed tensor store.

    Public data reads are canonical: :meth:`latest` returns ``(R,P,*data_shape)`` and
    :meth:`latest_tensor` returns the full typed envelope.  An operation that
    needs image/grid layout calls :meth:`latest_logical`, which uses the schema
    to restore ``(R,*point_shape,*data_shape)`` without inspecting ndarray
    rank.  History has a separate leading update axis and is exposed as
    :class:`TensorHistory` by :meth:`history_tensor`.
    """

    def __init__(self, *, history: int = DEFAULT_HISTORY):
        self._history_len = max(1, int(history))
        self._lock = threading.RLock()
        self._schemas: dict[str, SignalSchema] = {}
        self._schema_version: dict[str, int] = {}
        self._stores: dict[str, TensorStore] = {}
        self._history_limits: dict[str, int] = {}
        self._history_reservations: dict[str, dict[int, int]] = {}
        self._reservation_token = 0
        self._version = 0
        self._shot = 0
        self._source_shot = 0
        self._sig_version: dict[str, int] = {}
        self._cond = threading.Condition(self._lock)

    # ------------------------------------------------------------- schema side
    def _history_limit_for_locked(self, key: str) -> int:
        key = str(key)
        base = int(self._history_limits.get(key, self._history_len))
        reservations = self._history_reservations.get(key, {})
        return max([base, *reservations.values()]) if reservations else base

    def register_signal(
        self,
        name: str,
        schema: SignalSchema,
        *,
        history: int | None = None,
        replace: bool = False,
        initialize: bool = False,
        fill_value=None,
    ) -> SignalSchema:
        """Register the authoritative schema for ``name``.

        Re-registering an equal schema is idempotent.  A structural change is a
        new schema version and must be explicit (``replace=True``); this clears
        the old value/journal so values from two layouts can never share one
        history.  ``initialize=True`` preallocates a patchable store and requires
        a dtype plus ``repeat_capacity``.
        """

        if not isinstance(schema, SignalSchema):
            raise TypeError(f"schema must be SignalSchema, got {type(schema).__name__}.")
        key = str(name)
        with self._lock:
            current = self._schemas.get(key)
            if current is not None and current is not schema and not current.same_definition(schema):
                if not replace:
                    raise ValueError(
                        f"signal {key!r} is already registered with a different schema; "
                        "use replace=True to start an explicit new schema version.")
                version = self._schema_version.get(key, 1) + 1
                self._stores.pop(key, None)
                self._sig_version.pop(key, None)
            else:
                version = self._schema_version.get(key, 1)
            if history is not None:
                self._history_limits[key] = max(1, int(history))
            limit = self._history_limit_for_locked(key)
            self._schemas[key] = schema
            self._schema_version[key] = version
            store = self._stores.get(key)
            if store is None:
                store = TensorStore(schema, history=limit, schema_version=version,
                                    initialize=initialize, fill_value=fill_value)
                self._stores[key] = store
            else:
                store.set_history_limit(limit)
            return schema

    def configure_signal(self, name: str, *, history: int | None = None) -> None:
        """Set the retained forward update-journal depth for one signal."""

        key = str(name)
        with self._lock:
            if history is None:
                self._history_limits.pop(key, None)
            else:
                self._history_limits[key] = max(1, int(history))
            store = self._stores.get(key)
            if store is not None:
                store.set_history_limit(self._history_limit_for_locked(key))

    def reserve_history(self, names, capacity: int) -> HistoryReservation:
        """Reserve enough ordered updates for a finite consumer.

        Producer policy remains the baseline (for example an image stream keeps
        eight updates).  Active consumer reservations layer on top using ``max``;
        one consumer can never shrink another's or the producer's capacity.  The
        returned lease restores the effective baseline after release.
        """

        capacity = int(capacity)
        if capacity < 1:
            raise ValueError(f"history reservation capacity must be >= 1, got {capacity}.")
        selected = tuple(dict.fromkeys(str(name) for name in names))
        if not selected:
            raise ValueError("history reservation requires at least one signal name.")
        with self._lock:
            self._reservation_token += 1
            token = self._reservation_token
            for name in selected:
                self._history_reservations.setdefault(name, {})[token] = capacity
                store = self._stores.get(name)
                if store is not None:
                    store.set_history_limit(self._history_limit_for_locked(name))
        return HistoryReservation(self, token, selected, capacity)

    def _release_history_reservation(self, token: int, names) -> None:
        with self._lock:
            for name in names:
                reservations = self._history_reservations.get(str(name))
                if reservations is None:
                    continue
                reservations.pop(int(token), None)
                if not reservations:
                    self._history_reservations.pop(str(name), None)
                store = self._stores.get(str(name))
                if store is not None:
                    store.set_history_limit(self._history_limit_for_locked(str(name)))

    def schema(self, name: str) -> SignalSchema:
        with self._lock:
            try:
                return self._schemas[str(name)]
            except KeyError as exc:
                raise KeyError(
                    f"no schema registered for {name!r}; registered: {sorted(self._schemas)}") from exc

    def schemas(self) -> dict[str, SignalSchema]:
        with self._lock:
            return dict(self._schemas)

    def registered_names(self) -> list[str]:
        with self._lock:
            return sorted(self._schemas)

    def history_limit(self, name: str | None = None) -> int:
        with self._lock:
            if name is not None:
                return self._history_limit_for_locked(str(name))
            keys = {*self._history_limits, *self._history_reservations}
            if keys:
                return min([self._history_len,
                            *(self._history_limit_for_locked(key) for key in keys)])
            return self._history_len

    # ------------------------------------------------------------- publish side
    @staticmethod
    def _dtype_of(value) -> np.dtype:
        if isinstance(value, SignalTensor):
            return value.data.dtype
        return np.asarray(value).dtype

    def _prepare_full_locked(self, key: str, value):
        schema = self._schemas.get(key)
        is_new = schema is None
        if schema is None:
            tensor = value if isinstance(value, SignalTensor) else SignalTensor.from_external(value)
            schema = tensor.schema
            store = TensorStore(schema, history=self._history_limit_for_locked(key),
                                schema_version=self._schema_version.get(key, 1))
            return schema, store, tensor, is_new

        if schema.dtype is None:
            schema = schema.bind_dtype(self._dtype_of(value))
            # A dtype-less schema cannot have an initialized store.  Replace its
            # empty holder with the now fully-bound one before first publication.
            store = TensorStore(schema, history=self._history_limit_for_locked(key),
                                schema_version=self._schema_version.get(key, 1))
        else:
            store = self._stores[key]
        tensor = value if isinstance(value, SignalTensor) else SignalTensor.from_value(value, schema)
        schema.assert_compatible(tensor.schema, check_repeat=False)
        schema.physical_shape(tensor.data.shape[0])
        store.validate_tensor(tensor)
        return schema, store, tensor, is_new

    def publish(self, values: Mapping[str, object], *, shot: int | None = None,
                provenance: int | None = None) -> int:
        """Atomically normalize, validate and publish one acquisition update.

        Values are one of:

        * ``SignalTensor`` -- self-describing (and auto-registering when new);
        * ``TensorPatch`` -- an in-place update of an already registered store;
        * raw numeric value -- validated against a registered schema, or when
          unregistered normalized by the documented external-single-datum rule.

        Every full publish is preflighted before any store mutates.  Patches are
        preflighted likewise, so a multi-signal call cannot land only half its
        signals when one value has a bad shape or dtype.
        """

        if not isinstance(values, Mapping):
            raise TypeError("SignalHub.publish expects a mapping of name -> tensor/value/patch.")
        with self._lock:
            src_id = int(provenance) if provenance is not None else NO_LINEAGE
            if src_id < NO_LINEAGE:
                raise ValueError(
                    f"signal provenance must be {NO_LINEAGE} (no lineage) or a "
                    f"non-negative source-shot id; got {src_id}.")
            prepared = []
            for raw_name, value in values.items():
                key = str(raw_name)
                if isinstance(value, TensorPatch):
                    if key not in self._schemas or key not in self._stores:
                        raise KeyError(
                            f"cannot patch unregistered signal {key!r}; register its SignalSchema first.")
                    store = self._stores[key]
                    store.validate_patch(value)  # no data mutation; establishes shape/dtype validity
                    prepared.append((key, "patch", self._schemas[key], store, value, False))
                else:
                    schema, store, tensor, is_new = self._prepare_full_locked(key, value)
                    prepared.append((key, "full", schema, store, tensor, is_new))

            # All values are valid.  Install newly-bound schemas/stores, then
            # mutate the current tensors.  No normal validation error remains.
            for key, action, schema, store, payload, is_new in prepared:
                current = self._schemas.get(key)
                if is_new or current is None or (current is not schema and not current.same_definition(schema)) \
                        or self._stores.get(key) is not store:
                    self._schemas[key] = schema
                    self._schema_version.setdefault(key, 1)
                    self._stores[key] = store
                if action == "patch":
                    store.patch(payload, provenance=src_id)
                else:
                    store.publish(payload, provenance=src_id)
                self._sig_version[key] = self._sig_version.get(key, 0) + 1

            self._shot = int(shot) if shot is not None else self._shot + 1
            self._version += 1
            self._cond.notify_all()
            return self._version

    def publish_patch(self, name: str, patch: TensorPatch, *, shot: int | None = None,
                      provenance: int | None = None) -> int:
        return self.publish({str(name): patch}, shot=shot, provenance=provenance)

    def next_source_shot(self) -> int:
        with self._lock:
            self._source_shot += 1
            return self._source_shot

    def latest_provenance(self, name: str) -> int:
        with self._lock:
            store = self._stores.get(str(name))
            return int(store.provenance) if store is not None and store.version else NO_LINEAGE

    def provenance_map(self) -> dict[str, int]:
        with self._lock:
            return {name: int(store.provenance) for name, store in self._stores.items()
                    if store.version}

    def clear(self) -> None:
        with self._lock:
            # Keep generation tombstones.  A subscriber holding a cursor from
            # before clear() must never cross into a later same-named stream
            # merely because both stores happen to start their value versions
            # at one.  Schema generations are therefore monotonic for the
            # lifetime of the hub, including across removal and clear.
            generations = {
                name: max(1, int(version) + 1)
                for name, version in self._schema_version.items()
            }
            for name in {*self._schemas, *self._stores}:
                generations.setdefault(name, 2)
            self._schemas.clear()
            self._schema_version = generations
            self._stores.clear()
            self._sig_version.clear()
            self._history_limits.clear()
            self._history_reservations.clear()
            self._version += 1
            self._shot = 0
            self._cond.notify_all()

    def remove_signals(self, names) -> list[str]:
        """Remove values *and schemas* so a future same name cannot inherit stale layout."""

        removed = []
        with self._lock:
            for name in (str(n) for n in (names or [])):
                existed = name in self._schemas or name in self._stores
                self._schemas.pop(name, None)
                self._stores.pop(name, None)
                self._sig_version.pop(name, None)
                self._history_limits.pop(name, None)
                if existed:
                    # Preserve an incremented tombstone so an old cursor cannot
                    # silently enter a recreated stream with the same name and
                    # coincidentally matching per-store value version.
                    self._schema_version[name] = max(
                        1, int(self._schema_version.get(name, 1)) + 1)
                    removed.append(name)
            if removed:
                self._version += 1
                self._cond.notify_all()
        return removed

    def signal_versions(self) -> dict[str, int]:
        with self._lock:
            return dict(self._sig_version)

    def signal_cursors(self, names) -> SignalSubscription:
        """Create a stateful forward subscription after each current update.

        A missing signal starts at the wildcard ``(schema=0, version=0)`` and
        binds to its first future schema epoch.  Existing stores seed one private
        reconstructed state exactly once.  Subsequent reads advance that state
        through O(1)-addressed journal deltas; cursor mappings are handles back
        to this session, never stateless integer positions.
        """

        selected = tuple(dict.fromkeys(str(name) for name in names))
        with self._lock:
            states = {}
            for name in selected:
                store = self._stores.get(name)
                if store is None:
                    cursor, data, valid = SignalCursor(0, 0), None, None
                else:
                    cursor, data, valid = store.seed_for_cursor(store.cursor)
                states[name] = _SubscribedSignal(name, cursor, data, valid)
            return SignalSubscription(self, selected, states)

    def _resolve_subscription(
        self,
        selected: tuple[str, ...],
        cursors: Mapping[str, object] | SignalSubscription,
    ) -> SignalSubscription:
        missing = [name for name in selected if name not in cursors]
        extra = [str(name) for name in cursors if str(name) not in selected]
        if missing or extra:
            raise ValueError(
                f"cursor keys must exactly match names; missing={missing}, extra={extra}.")
        handles = [cursors[name] for name in selected]
        if not all(isinstance(handle, SignalSubscriptionCursor) for handle in handles):
            raise TypeError(
                "every cursor must be a SignalSubscriptionCursor from signal_cursors(); "
                "stateless SignalCursor replay is not supported.")
        sessions = {id(handle._subscription): handle._subscription for handle in handles}
        if len(sessions) != 1:
            raise ValueError("all cursor handles must belong to one coherent subscription session.")
        session = next(iter(sessions.values()))
        if session._hub is not self:
            raise ValueError("subscription cursor belongs to a different SignalHub.")
        if session.names != selected:
            raise ValueError(
                f"subscription name/order is {session.names}, requested {selected}; "
                "coherent event order is fixed when subscribing.")
        if any(handle.name != name for name, handle in zip(selected, handles)):
            raise ValueError("subscription cursor handles were assigned to the wrong signal names.")
        return session

    def _refresh_subscription_locked(self, session: SignalSubscription) -> None:
        newly_present: set[int] = set()
        for name in session.names:
            state = session._states[name]
            store = self._stores.get(name)
            if store is None:
                if state.schema_version != 0:
                    raise SignalHistoryGap(
                        f"signal {name!r} was removed after cursor {state.cursor}.")
                continue
            if state.schema_version == 0:
                state.schema_version = int(store.schema_version)
            elif state.schema_version != store.schema_version:
                raise SignalHistoryGap(
                    f"signal schema changed from version {state.schema_version} to "
                    f"{store.schema_version}; the old subscription cannot cross layouts.")

            if state.data is None:
                bound, data, valid = store.seed_for_cursor(state.cursor)
                state.schema_version = int(bound.schema_version)
                state.data, state.valid = data, valid
                if data is None:
                    continue
            else:
                # Validate the committed cursor, not only scan_version: pending
                # metadata never extends bounded payload retention.
                store.validate_cursor(state.cursor)

            scan = state.scan_cursor
            if len(session.names) == 1:
                # A single stream needs exactly its next delta.  Keep the
                # iterator lazy instead of mirroring an entire reserved backlog
                # as Python metadata inside the session.
                if not state.pending:
                    ref = store.next_update_ref(scan)
                    if ref is not None:
                        state.append(ref)
                continue
            while True:
                ref = store.next_update_ref(scan)
                if ref is None:
                    break
                if state.append(ref):
                    newly_present.add(int(ref.provenance))
                scan = ref.cursor

        if len(session.names) > 1:
            for provenance in newly_present:
                session._sync_common(provenance)

    def _advance_subscription_state_locked(
        self,
        session: SignalSubscription,
        name: str,
        target_version: int,
    ) -> tuple[SignalTensor, set[int]]:
        state = session._states[name]
        store = self._stores.get(name)
        if store is None:
            raise SignalHistoryGap(f"signal {name!r} disappeared while advancing its subscription.")
        if state.data is None or state.valid is None:
            raise SignalHistoryGap(f"signal {name!r} has no initialized state to advance.")
        affected: set[int] = set()
        selected_ref = None
        while state.pending:
            ref = state.pending.popleft()
            if ref.cursor.version > int(target_version):
                raise SignalHistoryGap(
                    f"subscription for {name!r} skipped target update {target_version}.")
            store.apply_update_to(state.data, state.valid, ref)
            state.version = int(ref.cursor.version)
            state.remove_pending_ref(ref)
            if ref.provenance != NO_LINEAGE:
                affected.add(int(ref.provenance))
            if state.version == int(target_version):
                selected_ref = ref
                break
        if selected_ref is None:
            raise SignalHistoryGap(
                f"subscription for {name!r} has no pending update {target_version}.")
        tensor = SignalTensor(
            state.data,
            store.schema,
            valid=state.valid,
            version=state.version,
            provenance=int(selected_ref.provenance),
            schema_version=store.schema_version,
        )
        return tensor, affected

    def _next_subscription_update_locked(
        self, session: SignalSubscription,
    ) -> CoherentUpdate | None:
        self._refresh_subscription_locked(session)
        if len(session.names) == 1:
            name = session.names[0]
            state = session._states[name]
            if not state.pending:
                return None
            ref = state.pending[0]
            tensor, _ = self._advance_subscription_state_locked(
                session, name, ref.cursor.version)
            return CoherentUpdate(
                provenance=int(ref.provenance),
                tensors={name: tensor},
                cursors=session,
            )

        provenance = session._next_common()
        if provenance is None:
            return None
        targets = {
            name: session._states[name].first_version(provenance)
            for name in session.names
        }
        if any(version is None for version in targets.values()):
            session._sync_common(provenance, refresh_position=True)
            return None
        tensors = {}
        affected: set[int] = set()
        for name in session.names:
            tensor, changed = self._advance_subscription_state_locked(
                session, name, int(targets[name]))
            tensors[name] = tensor
            affected.update(changed)
        for changed_provenance in affected:
            session._sync_common(changed_provenance, refresh_position=True)
        return CoherentUpdate(
            provenance=int(provenance),
            tensors=tensors,
            cursors=session,
        )

    def next_coherent_update(
        self,
        names,
        cursors: Mapping[str, object] | SignalSubscription | None = None,
        *,
        timeout: float | None = None,
        stop=None,
    ) -> CoherentUpdate | None:
        """Return the next update from a stateful forward subscription.

        For one name this is simply its next journal entry, even when it has no
        physical lineage.  For multiple names, the method waits until every
        stream contains an entry with the same non-negative provenance and then
        advances each exact reconstructed state to that entry.  Faster streams may have
        unmatched updates before the shared lineage; the returned cursors advance
        past those records deliberately.

        Journals are bounded.  If any first-unconsumed event was evicted, or a
        schema version changed under the subscriber, ``SignalHistoryGap`` is
        raised immediately.  The method never falls forward to ``latest``.
        ``None`` means timeout or cooperative stop.
        """

        selected = tuple(dict.fromkeys(str(name) for name in names))
        if not selected:
            raise ValueError("next_coherent_update requires at least one signal name.")
        session = self.signal_cursors(selected) if cursors is None \
            else self._resolve_subscription(selected, cursors)

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            while True:
                update = self._next_subscription_update_locked(session)
                if update is not None:
                    return update

                if stop is not None and stop.is_set():
                    return None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                else:
                    remaining = None
                wait_for = remaining
                if stop is not None:
                    wait_for = 0.05 if remaining is None else min(remaining, 0.05)
                self._cond.wait(wait_for)

    # ------------------------------------------------------------- wait/clock
    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def wait_for_change(self, since_version: int, *, timeout: float, stop=None) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            while self._version <= since_version:
                if stop is not None and stop.is_set():
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(min(remaining, 0.05) if stop is not None else remaining)
            return True

    @property
    def shot(self) -> int:
        with self._lock:
            return self._shot

    @property
    def history_len(self) -> int:
        return self.history_limit()

    # ------------------------------------------------------------- consumer side
    def names(self) -> list[str]:
        """Published signal names (registered-but-empty schemas are excluded)."""

        with self._lock:
            return sorted(name for name, store in self._stores.items() if store.version)

    def _store_with_value_locked(self, name: str) -> TensorStore:
        key = str(name)
        store = self._stores.get(key)
        if store is None or not store.version:
            raise KeyError(f"no signal named {name!r}; available: {self.names()}")
        return store

    def latest_tensor(self, name: str) -> SignalTensor:
        with self._lock:
            return self._store_with_value_locked(name).snapshot()

    def latest(self, name: str) -> np.ndarray:
        """Latest canonical ``(R, P, *data_shape)`` ndarray copy."""

        return self.latest_tensor(name).data.copy()

    def latest_logical(self, name: str) -> np.ndarray:
        """Latest ``(R,*point_shape,*data_shape)`` value from schema metadata."""

        return self.latest_tensor(name).logical()

    def latest_valid(self, name: str) -> np.ndarray:
        return self.latest_tensor(name).valid.copy()

    def history_tensor(self, name: str, n: int | None = None) -> TensorHistory:
        with self._lock:
            return self._store_with_value_locked(name).history(n)

    def history(self, name: str, n: int | None = None) -> np.ndarray:
        """Canonical history ``(updates, R, P, *data_shape)``."""

        return self.history_tensor(name, n).data.copy()

    def history_logical(self, name: str, n: int | None = None) -> np.ndarray:
        return self.history_tensor(name, n).logical()

    def snapshot_latest(self, names=None, *, tensors: bool = False,
                        logical: bool = False) -> dict[str, object]:
        with self._lock:
            selected = self.names() if names is None else [str(n) for n in names]
            out = {}
            for name in selected:
                store = self._stores.get(name)
                if store is None or not store.version:
                    continue
                tensor = store.snapshot()
                out[name] = tensor if tensors else (tensor.logical() if logical else tensor.data.copy())
            return out

    def snapshot_at(self, target: int | None, names=None, *, tensors: bool = False,
                    logical: bool = False) -> dict[str, object]:
        """Shot-coherent snapshot reconstructed from each store's patch journal."""

        with self._lock:
            selected = self.names() if names is None else [str(n) for n in names]
            out = {}
            for name in selected:
                store = self._stores.get(name)
                if store is None or not store.version:
                    continue
                tensor = store.snapshot()
                if target is not None and tensor.provenance != NO_LINEAGE:
                    selected_tensor = store.at_provenance(int(target))
                    if selected_tensor is None:
                        raise SignalHistoryGap(
                            f"signal {name!r} has no retained state for provenance {int(target)}; "
                            "a shot-coherent snapshot cannot substitute its latest value.")
                    tensor = selected_tensor
                out[name] = tensor if tensors else (tensor.logical() if logical else tensor.data.copy())
            return out

    def storage_stats(self, name: str) -> dict[str, int]:
        with self._lock:
            store = self._stores.get(str(name))
            if store is None:
                raise KeyError(f"no store for signal {name!r}")
            return store.stats()


__all__ = [
    "CoherentUpdate",
    "DEFAULT_HISTORY",
    "HistoryReservation",
    "NO_LINEAGE",
    "SignalCursor",
    "SignalHistoryGap",
    "SignalHub",
    "SignalSchema",
    "SignalSubscription",
    "SignalSubscriptionCursor",
    "SignalTensor",
    "TensorHistory",
    "TensorPatch",
]
