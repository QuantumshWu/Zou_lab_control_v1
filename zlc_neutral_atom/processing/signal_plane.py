"""Headless signal plane with producer-local causal coherence.

A hosted producer publishes through a ``LiveDatasetPort``; this plane freezes
only sources that reported a new revision. Each source is one producer
transaction. Combining their latest immutable fronts for one consumer cycle
does not assert that independent producers observed the same physical event.

Freeze-latest, not a bus.  Each changed slot materialises its own atomic
transaction exactly once; unchanged slots reuse their immutable fronts.  Independent
producers still advance independently.  Within one explicit source -> Processor
component, however, a newer source and its active descendants replace the previous
component together. A slow Processor therefore cannot expose source revision N
beside its own derived revision N-1.

What replaced the shot clock: a monitor tap overwrites when its consumer falls
behind rather than back-pressuring acquisition, and it says so per signal
through :class:`~zlc_neutral_atom.runtime.dataset.MonitorCoverage`
(written/total cells, missed events, current gap).  There is no global shot
counter to compare against, and reintroducing one would be a fiction -- signals
from different runs advance independently.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, runtime_checkable
import uuid
from weakref import WeakKeyDictionary

from zlc_data import (
    BlockId,
    DataBlock,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    StreamGenerationId,
    ValueSchema,
)
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCoverage,
    MonitorCoverage,
)
from zlc_neutral_atom.runtime.signal_source import (
    SignalEventAssociationSource,
    SignalEventSource,
)
from zlc_neutral_atom.runtime.streams import EventRef, StreamId
from zlc_storage import canonical_text

__all__ = [
    "DerivedSignalOutput",
    "LatestProcessorControl",
    "SignalDataPlane",
    "SignalFront",
    "SignalPublication",
    "SignalProducer",
    "SignalValue",
]


@runtime_checkable
class SignalProducer(Protocol):
    """Stable routing contract implemented by a hosted producer.

    The plane receives an immutable producer identity and owner declarations;
    it never discovers behavior from a registry or uses Python object identity
    as a routing key.
    """

    instance_id: str
    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]: ...

    def signal_key(self, output_name: str) -> str: ...


@runtime_checkable
class LatestProcessorControl(SignalProducer, Protocol):
    """Public owner callbacks used by the shared latest-only lane."""

    def prepare_processor_application(self) -> object: ...

    def validate_processor_source(self, source: "SignalValue") -> None: ...

    def processor_application_ready(self, application: object) -> None: ...

    def processor_work_started(self, source: "SignalValue") -> None: ...

    def accept_processor_result(
        self,
        source: "SignalValue",
        source_publication: "SignalPublication",
        result: object,
    ) -> None: ...

    def accept_processor_failure(self, error: Exception) -> None: ...

    def accept_processor_cancelled(self) -> None: ...

    def request_processor_owner_wake(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DerivedSignalOutput:
    """One consumer-derived immutable value without presentation metadata."""

    snapshot: OwnedSnapshot
    preserve_source_coverage: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("derived signal snapshot must be OwnedSnapshot")
        if type(self.preserve_source_coverage) is not bool:
            raise TypeError("preserve_source_coverage must be bool")


@dataclass(frozen=True)
class SignalValue:
    """One signal at one producer-owned immutable revision."""

    name: str
    snapshot: OwnedSnapshot
    coverage: DatasetCoverage | MonitorCoverage | None
    transient: bool = False         # withdrawn with its live producer

    def __post_init__(self) -> None:
        name = canonical_text(self.name, "signal name")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("signal snapshot must be OwnedSnapshot")
        if self.coverage is not None and not isinstance(
            self.coverage,
            (DatasetCoverage, MonitorCoverage),
        ):
            raise TypeError("signal coverage has an unknown type")
        if not isinstance(self.transient, bool):
            raise TypeError("signal transient flag must be bool")
        object.__setattr__(self, "name", name)

    # The block is the value; these read off it rather than copying, so two
    # consumers describing "the same signal" cannot describe different data.
    @property
    def block(self):
        """The snapshot's DataBlock -- shape/dtype/schema live here."""

        return self.snapshot.block

    @property
    def schema(self):
        return self.block.schema

    @property
    def values(self):
        """The block's array.  Read-only by ownership: never mutate a frozen block."""

        return self.block.values

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    @property
    def cell_schema(self):
        """The per-cell value schema -- where dtype / unit / data axes live.

        A DatasetSchema describes the DATASET (repeat axis, point axes, layout);
        what a cell actually holds is its ``cell_schema``.  Reading through it
        keeps every consumer on the description the producer declared.
        """

        return self.schema.cell_schema

    @property
    def dtype(self):
        return self.cell_schema.dtype

    @property
    def unit(self) -> str | None:
        return self.cell_schema.value_unit

    @property
    def axes(self) -> tuple:
        return tuple(self.cell_schema.data_axes)

    @property
    def behind(self) -> int:
        """How many events the tap dropped for this signal -- 0 when keeping up.

        This is what consumer-lag telemetry reads.  It is per signal, from
        the tap that actually dropped them; there is no global shot counter to
        subtract, and inventing one would mean comparing runs that advance
        independently.
        """

        if isinstance(self.coverage, MonitorCoverage):
            return self.coverage.missed_events
        return 0


@dataclass(frozen=True, eq=False)
class SignalPublication:
    """One exact immutable signal transaction.

    A publication is the causal unit.  Its signal mapping is one atomic sibling
    bundle and ``direct_parent_refs`` names only the exact events consumed to
    produce it.  The plane privately retains the corresponding immutable parent
    payloads while a child is live; that process-local retention is deliberately
    not part of the public lineage contract.  Run identity, configuration and
    domain provenance remain with their Run/domain records rather than being
    mirrored into every signal event.
    """

    event_ref: EventRef
    signals: Mapping[str, SignalValue]
    _issuer: object = field(repr=False, compare=False)
    direct_parent_refs: tuple[EventRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.event_ref, EventRef):
            raise TypeError("signal publication event_ref must be EventRef")
        signals = dict(self.signals)
        if not signals:
            raise ValueError("signal publication requires an atomic sibling bundle")
        if any(
            not isinstance(name, str)
            or not isinstance(value, SignalValue)
            or value.name != name
            for name, value in signals.items()
        ):
            raise TypeError("signal publication mapping differs from its SignalValues")
        parents = tuple(self.direct_parent_refs)
        if any(not isinstance(value, EventRef) for value in parents):
            raise TypeError("signal publication parents must be EventRef values")
        if len(set(parents)) != len(parents):
            raise ValueError("signal publication parent refs must be unique")
        object.__setattr__(self, "signals", MappingProxyType(signals))
        object.__setattr__(self, "direct_parent_refs", parents)

    def value(self, name: str) -> SignalValue | None:
        return self.signals.get(str(name))


@dataclass(frozen=True)
class SignalFront:
    """Immutable front: coherent derived components, independent producers."""

    signals: Mapping[str, SignalValue]
    failures: Mapping[str, str]     # producer instance_id -> freeze failure
    publication_by_signal: Mapping[str, SignalPublication] = field(
        default_factory=dict,
        repr=False,
    )
    _continuous_group_by_signal: Mapping[str, frozenset[str]] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        signals = dict(self.signals)
        failures = dict(self.failures)
        publications = dict(self.publication_by_signal)
        groups = {
            str(name): frozenset(members)
            for name, members in self._continuous_group_by_signal.items()
        }
        if any(not isinstance(value, SignalValue) for value in signals.values()):
            raise TypeError("SignalFront signals must contain SignalValue values")
        if any(not isinstance(value, str) for value in failures.values()):
            raise TypeError("SignalFront failures must contain strings")
        if set(publications) != set(signals):
            raise ValueError("SignalFront publications must cover every signal")
        for name, publication in publications.items():
            if not isinstance(publication, SignalPublication):
                raise TypeError("SignalFront contains another publication type")
            if publication.value(name) is not signals[name]:
                raise ValueError(
                    "SignalFront value is not owned by its exact publication"
                )
        for name, members in groups.items():
            if name not in signals or name not in members:
                raise ValueError("continuous group must contain its visible signal")
            if not members or not members.issubset(signals):
                raise ValueError("continuous group contains a non-visible signal")
            if any(groups.get(member) != members for member in members):
                raise ValueError("continuous signal groups must be symmetric")
        object.__setattr__(self, "signals", MappingProxyType(signals))
        object.__setattr__(self, "failures", MappingProxyType(failures))
        object.__setattr__(
            self,
            "publication_by_signal",
            MappingProxyType(publications),
        )
        object.__setattr__(
            self,
            "_continuous_group_by_signal",
            MappingProxyType(groups),
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self.signals)

    def value(self, name: str) -> SignalValue | None:
        return self.signals.get(str(name))

    def publication(self, name: str) -> SignalPublication | None:
        return self.publication_by_signal.get(str(name))

    def continuous_group(self, name: str) -> frozenset[str]:
        """Return the already-resolved visible causal group for one signal."""

        selected = str(name)
        if selected not in self.signals:
            return frozenset()
        return self._continuous_group_by_signal.get(
            selected,
            frozenset((selected,)),
        )


def _declared_outputs(declarations) -> dict[str, DatasetOutputDeclaration]:
    """Return one producer's frozen Dataset output declarations."""

    values = tuple(declarations)
    if any(not isinstance(value, DatasetOutputDeclaration) for value in values):
        raise TypeError(
            "signal outputs must retain DatasetOutputDeclaration values"
        )
    result = {value.name: value for value in values}
    if len(result) != len(values):
        raise ValueError("signal output declarations contain duplicate names")
    return result


def _require_published_declaration(
    route_name: str,
    output: FinalDatasetOutput | LiveDatasetOutput,
    declared: Mapping[str, DatasetOutputDeclaration],
) -> None:
    if output.name != route_name:
        raise ValueError("published output key differs from its owner declaration")
    expected = declared.get(route_name)
    if expected is None:
        raise ValueError(
            "published output is absent from the frozen producer vocabulary"
        )
    if output.declaration != expected:
        raise ValueError(
            "published output contract differs from the frozen owner declaration"
        )


def _require_signal_producer(node: object) -> SignalProducer:
    if not isinstance(node, SignalProducer):
        raise TypeError("signal producer must implement SignalProducer")
    return node


def _node_instance_id(node: object) -> str:
    """Return the stable producer identity required by the signal plane."""

    producer = _require_signal_producer(node)
    return canonical_text(producer.instance_id, "signal producer instance_id")


def _evaluate_prepared_processor_application(
    application: object,
    source: SignalValue,
    coverage: MonitorCoverage,
) -> object:
    """Run one domain-owned Processor operation over an admitted revision.

    The data plane knows only the common application seam: an immutable source,
    typed coverage go into the already-prepared domain
    command.  It never reconstructs Camera/Occupancy shapes or output schemas.
    """

    evaluate = getattr(application, "evaluate", None)
    if not callable(evaluate):
        raise TypeError(
            "prepared Processor application must expose evaluate()"
        )
    result = evaluate(
        source.snapshot,
        coverage,
    )
    from zlc_neutral_atom.processing.causal import (
        require_causal_processor_evaluation,
    )

    return require_causal_processor_evaluation(result)


@dataclass(slots=True)
class _GenerationState:
    """The sole mutable state for one process-local signal generation."""

    owner_id: str
    generation: StreamGenerationId
    kind: str
    output_names: tuple[str, ...]
    bare_names: Mapping[str, str]
    node: object | None = None
    slot: object | None = None
    source_name: str | None = None
    source_owner_id: str | None = None
    source_generation: StreamGenerationId | None = None
    publication: SignalPublication | None = None
    event_source: SignalEventSource | None = None
    event_output_name: str | None = None
    bound_parents: tuple[SignalPublication, ...] = ()
    last_parent_sequence: int = 0
    published_names: tuple[str, ...] | None = None
    published_schemas: Mapping[str, DatasetSchema] | None = None
    next_sequence: int = 1
    failure: str | None = None
    terminal: bool = False
    retired: bool = False


@dataclass(slots=True)
class _ProcessorEntry:
    node: LatestProcessorControl
    source_name: str
    prepare_future: Future
    application: object | None = None
    work_future: Future | None = None
    work_publication: SignalPublication | None = None
    pending_publication: SignalPublication | None = None
    last_publication: SignalPublication | None = None
    cancel_requested: bool = False


class _LatestOnlyProcessorLane:
    """One private latest-only execution seam for reactive Processors."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="signal-latest-processor",
        )
        self._processors: dict[str, _ProcessorEntry] = {}
        self._closed = False

    @staticmethod
    def _require_node(node: object) -> LatestProcessorControl:
        if not isinstance(node, LatestProcessorControl):
            raise TypeError("Processor must implement LatestProcessorControl")
        return node

    def attach_processor(
        self,
        node: LatestProcessorControl,
        source_name: str,
        initial_publication: SignalPublication,
    ) -> None:
        if self._closed:
            raise RuntimeError("continuous worker lane is closed")
        node = self._require_node(node)
        name = canonical_text(source_name, "processor source name")
        source = initial_publication.value(name)
        if source is None:
            raise ValueError("initial publication has no selected Processor source")
        key = _node_instance_id(node)
        if key in self._processors:
            raise RuntimeError("Processor node is already attached")
        node.validate_processor_source(source)
        future = self._executor.submit(node.prepare_processor_application)
        future.add_done_callback(
            lambda _future, current=node: self._wake_processor(current)
        )
        self._processors[key] = _ProcessorEntry(
            node=node,
            source_name=name,
            prepare_future=future,
            pending_publication=initial_publication,
            last_publication=initial_publication,
        )

    def route(self, publications: Mapping[str, SignalPublication]) -> None:
        if self._closed:
            return
        for entry in tuple(self._processors.values()):
            if entry.cancel_requested:
                continue
            publication = publications.get(entry.source_name)
            if publication is None or publication is entry.last_publication:
                continue
            source = publication.value(entry.source_name)
            if source is None:
                self._fail_processor(
                    entry,
                    RuntimeError("Processor publication lost its selected signal"),
                )
                continue
            try:
                entry.node.validate_processor_source(source)
            except Exception as error:
                self._fail_processor(entry, error)
                continue
            entry.last_publication = publication
            entry.pending_publication = publication
            self._start_processor(entry)

    def drain_processors(self) -> None:
        for entry in tuple(self._processors.values()):
            if not entry.prepare_future.done():
                continue
            if entry.cancel_requested and entry.work_future is None:
                self._cancelled_processor(entry)
                continue
            if entry.application is None:
                try:
                    application = entry.prepare_future.result()
                    entry.node.processor_application_ready(application)
                except Exception as error:
                    self._fail_processor(entry, error)
                    continue
                entry.application = application
            work = entry.work_future
            if work is not None and work.done():
                publication = entry.work_publication
                entry.work_future = None
                entry.work_publication = None
                if entry.cancel_requested:
                    self._cancelled_processor(entry)
                    continue
                try:
                    if publication is None:
                        raise RuntimeError("Processor lane lost its exact publication")
                    source = publication.value(entry.source_name)
                    if source is None:
                        raise RuntimeError("Processor publication lost its source")
                    result = work.result()
                    entry.node.accept_processor_result(
                        source,
                        publication,
                        result,
                    )
                except Exception as error:
                    self._fail_processor(entry, error)
                    continue
            if entry.cancel_requested:
                if entry.work_future is None:
                    self._cancelled_processor(entry)
                continue
            self._start_processor(entry)

    def cancel_processor(self, node: object) -> bool:
        entry = self._processors.get(_node_instance_id(node))
        if entry is None:
            return True
        entry.cancel_requested = True
        entry.pending_publication = None
        idle = entry.prepare_future.done() and (
            entry.work_future is None or entry.work_future.done()
        )
        if idle:
            self._cancelled_processor(entry)
        return idle

    def active_processor_bindings(self) -> tuple[tuple[object, str], ...]:
        return tuple(
            (entry.node, entry.source_name)
            for entry in self._processors.values()
            if not entry.cancel_requested
        )

    def _start_processor(self, entry: _ProcessorEntry) -> None:
        if (
            entry.cancel_requested
            or entry.application is None
            or entry.work_future is not None
            or entry.pending_publication is None
        ):
            return
        publication, entry.pending_publication = entry.pending_publication, None
        source = publication.value(entry.source_name)
        try:
            if source is None:
                raise RuntimeError("Processor pending publication lost its source")
            coverage = source.coverage
            if not isinstance(coverage, MonitorCoverage):
                raise TypeError("latest Processor source requires MonitorCoverage")
            entry.node.processor_work_started(source)
            future = self._executor.submit(
                _evaluate_prepared_processor_application,
                entry.application,
                source,
                coverage,
            )
        except Exception as error:
            self._fail_processor(entry, error)
            return
        entry.work_publication = publication
        entry.work_future = future
        future.add_done_callback(
            lambda _future, current=entry.node: self._wake_processor(current)
        )

    def _fail_processor(self, entry: _ProcessorEntry, error: Exception) -> None:
        self._processors.pop(_node_instance_id(entry.node), None)
        entry.node.accept_processor_failure(error)

    def _cancelled_processor(self, entry: _ProcessorEntry) -> None:
        self._processors.pop(_node_instance_id(entry.node), None)
        entry.node.accept_processor_cancelled()

    def _wake_processor(self, node: object) -> None:
        if not self._closed:
            node.request_processor_owner_wake()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for entry in self._processors.values():
            entry.cancel_requested = True
            entry.pending_publication = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._processors.clear()


class SignalDataPlane:
    """One owner for signal generations, publications, and visible frontiers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lane = _LatestOnlyProcessorLane()
        self._publication_issuer = object()
        self._publication_parents: WeakKeyDictionary[
            SignalPublication,
            tuple[SignalPublication, ...],
        ] = WeakKeyDictionary()
        self._states: dict[str, _GenerationState] = {}
        self._dirty: set[str] = set()
        self._front_signals: frozenset[str] = frozenset()
        self._membership_changed = False
        self._closed = False
        self._request_owner_wake: Callable[[], None] | None = None
        self._owner_wake_token: object | None = None
        self._front = SignalFront({}, {})

    def bind_owner_wake(self, request_owner_wake: Callable[[], None]) -> object:
        if not callable(request_owner_wake):
            raise TypeError("request_owner_wake must be callable")
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            if self._request_owner_wake is not None:
                raise RuntimeError("signal data plane owner wake is already bound")
            token = object()
            self._request_owner_wake = request_owner_wake
            self._owner_wake_token = token
            return token

    def unbind_owner_wake(self, token: object) -> None:
        """Release only the exact Workbench wake borrow that was admitted."""

        with self._lock:
            if token is not self._owner_wake_token:
                raise RuntimeError("signal data plane owner wake token is not current")
            self._request_owner_wake = None
            self._owner_wake_token = None

    def set_front_signals(self, signal_names) -> None:
        """Set the connected continuous signal set whose front must be coherent."""

        names = frozenset(
            canonical_text(name, "connected signal name")
            for name in signal_names
        )
        wake = None
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            if names != self._front_signals:
                self._front_signals = names
                self._membership_changed = True
                wake = self._request_owner_wake
        if wake is not None:
            wake()

    @staticmethod
    def _mint_generation_locked() -> StreamGenerationId:
        return StreamGenerationId(uuid.uuid4().hex)

    def _state_for_signal_locked(
        self,
        signal_name: str,
    ) -> _GenerationState | None:
        selected = None
        for state in self._states.values():
            if state.retired or signal_name not in state.output_names:
                continue
            if selected is not None:
                raise RuntimeError(
                    f"signal {signal_name!r} has more than one generation owner"
                )
            selected = state
        return selected

    def _assert_names_available_locked(
        self,
        owner_id: str,
        output_names: tuple[str, ...],
    ) -> None:
        for name in output_names:
            state = self._state_for_signal_locked(name)
            if state is not None and state.owner_id != owner_id:
                raise RuntimeError(
                    f"signal {name!r} is already owned by {state.owner_id!r}"
                )

    @staticmethod
    def _generation_ref(
        state: _GenerationState,
    ) -> tuple[str, StreamGenerationId]:
        return state.owner_id, state.generation

    def _state_for_generation_ref_locked(
        self,
        reference: tuple[str, StreamGenerationId],
        *,
        allow_retired: bool = False,
    ) -> _GenerationState:
        if (
            not isinstance(reference, tuple)
            or len(reference) != 2
            or not isinstance(reference[0], str)
            or not isinstance(reference[1], StreamGenerationId)
        ):
            raise TypeError(
                "generation reference must contain owner_id and StreamGenerationId"
            )
        owner_id = canonical_text(reference[0], "signal generation owner_id")
        state = self._states.get(owner_id)
        if (
            state is None
            or state.generation != reference[1]
            or (state.retired and not allow_retired)
        ):
            raise RuntimeError("signal generation is no longer active")
        return state

    def _drop_state_locked(self, state: _GenerationState) -> None:
        if self._states.get(state.owner_id) is not state:
            return
        self._states.pop(state.owner_id)
        state.retired = True
        state.publication = None
        state.failure = None
        self._dirty.discard(state.owner_id)

    def _install_state_locked(
        self,
        *,
        owner_id: str,
        kind: str,
        output_names: tuple[str, ...],
        bare_names: Mapping[str, str],
        node: object | None = None,
        slot: object | None = None,
        source_name: str | None = None,
        event_source: SignalEventSource | None = None,
        event_output_name: str | None = None,
    ) -> _GenerationState:
        identity = canonical_text(owner_id, "signal generation owner_id")
        kind = canonical_text(kind, "signal generation kind")
        names = tuple(
            canonical_text(name, "signal generation output name")
            for name in output_names
        )
        if not names or len(set(names)) != len(names):
            raise ValueError("signal generation outputs must be non-empty and unique")
        if identity in self._states:
            raise RuntimeError("signal generation owner is already active")
        bare = dict(bare_names)
        if set(bare) != set(names):
            raise ValueError("signal generation bare names differ from its outputs")
        if any(
            not isinstance(value, str) or not value or value.strip() != value
            for value in bare.values()
        ):
            raise ValueError("signal generation bare names must be canonical text")
        source_state = None
        if source_name is not None:
            source_name = canonical_text(source_name, "signal route source name")
            source_state = self._state_for_signal_locked(source_name)
            if source_state is None:
                raise LookupError(
                    f"signal route source {source_name!r} is not active"
                )
        if event_source is not None and not isinstance(
            event_source,
            SignalEventSource,
        ):
            raise TypeError("event_source must implement SignalEventSource")
        if (event_source is None) != (event_output_name is None):
            raise ValueError(
                "event_source and event_output_name must be supplied together"
            )
        if event_output_name is not None:
            event_output_name = canonical_text(
                event_output_name,
                "signal event output name",
            )
        self._assert_names_available_locked(identity, names)
        state = _GenerationState(
            owner_id=identity,
            generation=self._mint_generation_locked(),
            kind=kind,
            output_names=names,
            bare_names=MappingProxyType(bare),
            node=node,
            slot=slot,
            source_name=source_name,
            source_owner_id=(
                None if source_state is None else source_state.owner_id
            ),
            source_generation=(
                None if source_state is None else source_state.generation
            ),
            event_source=event_source,
            event_output_name=event_output_name,
        )
        self._states[identity] = state
        self._membership_changed = True
        return state

    @staticmethod
    def _node_route_names(node: object) -> tuple[
        tuple[str, ...],
        Mapping[str, str],
    ]:
        producer = _require_signal_producer(node)
        declarations = _declared_outputs(producer.dataset_output_declarations)
        qualified = tuple(str(producer.signal_key(name)) for name in declarations)
        return (
            qualified,
            MappingProxyType(
                {
                    selected: bare
                    for selected, bare in zip(
                        qualified,
                        declarations,
                        strict=True,
                    )
                }
            ),
        )

    def reserve(self, node: object) -> StreamGenerationId:
        """Reserve one producer generation before its worker can publish.

        The composition owner performs this before submitting a run.  A later
        live attachment upgrades the same state in place; a FINAL-only run uses
        it directly.  Publication can therefore never recreate a generation
        after retirement.
        """

        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            existing = self._states.get(owner_id)
            if existing is not None:
                if (
                    not existing.retired
                    and not existing.terminal
                    and existing.kind == "producer"
                    and existing.node is node
                    and existing.output_names == output_names
                    and dict(existing.bare_names) == dict(bare_names)
                    and existing.slot is None
                ):
                    return existing.generation
                raise RuntimeError("producer generation is already active")
            state = self._install_state_locked(
                owner_id=owner_id,
                kind="producer",
                output_names=output_names,
                bare_names=bare_names,
                node=node,
            )
            return state.generation

    def require_active_generation(
        self,
        reference: tuple[str, StreamGenerationId],
    ) -> None:
        """Require one exact signal generation immediately before Run start."""

        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            self._state_for_generation_ref_locked(reference)

    def bind_generation_source(
        self,
        node: object,
        *,
        source_name: str,
        expected_generation: StreamGenerationId,
    ) -> None:
        """Bind one reserved producer to its exact live signal dependency."""

        source_name = canonical_text(source_name, "signal dependency source")
        if not isinstance(expected_generation, StreamGenerationId):
            raise TypeError("expected_generation must be StreamGenerationId")
        owner_id = _node_instance_id(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            state = self._states.get(owner_id)
            source = self._state_for_signal_locked(source_name)
            if (
                state is None
                or state.retired
                or state.kind != "producer"
                or state.node is not node
                or state.slot is not None
                or state.publication is not None
            ):
                raise RuntimeError(
                    "signal dependency must be bound to a reserved producer"
                )
            if state.source_name is not None:
                raise RuntimeError("producer signal dependency is already bound")
            if (
                source is None
                or source.retired
                or source.terminal
                or source.generation != expected_generation
            ):
                raise RuntimeError("signal dependency generation is not active")
            if source is state:
                raise ValueError("a signal generation cannot depend on itself")
            ancestor = source
            seen = set()
            while ancestor.source_owner_id is not None:
                reference = self._generation_ref(ancestor)
                if reference in seen:
                    raise RuntimeError("signal dependency graph contains a cycle")
                seen.add(reference)
                if ancestor.source_owner_id == state.owner_id:
                    raise ValueError("signal dependency would create a cycle")
                candidate = self._states.get(ancestor.source_owner_id)
                if (
                    candidate is None
                    or candidate.generation != ancestor.source_generation
                ):
                    break
                ancestor = candidate
            state.source_name = source_name
            state.source_owner_id = source.owner_id
            state.source_generation = source.generation
            self._membership_changed = True

    def release_generation_source(
        self,
        reference: tuple[str, StreamGenerationId],
    ) -> bool:
        """Release one terminal Run's live-input dependency, retaining its FINAL."""

        with self._lock:
            try:
                state = self._state_for_generation_ref_locked(reference)
            except RuntimeError:
                return False
            if state.source_owner_id is None:
                return False
            state.source_name = None
            state.source_owner_id = None
            state.source_generation = None
            self._membership_changed = True
            return True

    def withdraw_dependency_closure(
        self,
        roots: tuple[tuple[str, StreamGenerationId], ...],
        withdrawable_producer_refs: tuple[
            tuple[str, StreamGenerationId], ...
        ],
    ) -> tuple[tuple[str, StreamGenerationId], ...] | None:
        """Atomically withdraw one exact dependency closure, or mutate nothing.

        The Experiment application decides which admitted producer Runs are
        preemptible and supplies that exact allow-list.  This plane knows no
        RunId or policy; it only closes the signal graph.  Slots remain open
        until physical cleanup and lease release complete.
        """

        if not isinstance(roots, tuple):
            raise TypeError("dependency roots must be a tuple")
        if not isinstance(withdrawable_producer_refs, tuple):
            raise TypeError("withdrawable producer refs must be a tuple")
        if not roots or len(set(roots)) != len(roots):
            raise ValueError("dependency roots must be non-empty and unique")
        if len(set(withdrawable_producer_refs)) != len(
            withdrawable_producer_refs
        ):
            raise ValueError("withdrawable producer refs must be unique")
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            try:
                root_states = tuple(
                    self._state_for_generation_ref_locked(reference)
                    for reference in roots
                )
                withdrawable = {
                    self._generation_ref(
                        self._state_for_generation_ref_locked(reference)
                    )
                    for reference in withdrawable_producer_refs
                }
            except RuntimeError:
                return None
            if any(state.kind != "producer" for state in root_states):
                raise ValueError("dependency retirement roots must be producers")
            selected = {
                self._generation_ref(state)
                for state in root_states
            }
            changed = True
            while changed:
                changed = False
                for state in self._states.values():
                    reference = self._generation_ref(state)
                    source_ref = (
                        None
                        if state.source_owner_id is None
                        else (state.source_owner_id, state.source_generation)
                    )
                    if (
                        state.retired
                        or reference in selected
                        or source_ref not in selected
                    ):
                        continue
                    selected.add(reference)
                    changed = True
            states = tuple(
                state
                for state in self._states.values()
                if self._generation_ref(state) in selected
            )
            if any(
                state.kind == "producer"
                and self._generation_ref(state) not in withdrawable
                for state in states
            ):
                return None
            for state in states:
                state.retired = True
                state.publication = None
                state.failure = None
                self._dirty.discard(state.owner_id)
            self._membership_changed = True
            return tuple(self._generation_ref(state) for state in states)

    def finish_dependency_retirement(
        self,
        references: tuple[tuple[str, StreamGenerationId], ...],
    ) -> tuple[BaseException, ...]:
        """Close a withdrawn signal closure after its Runs released hardware."""

        if not isinstance(references, tuple):
            raise TypeError("retired generation refs must be a tuple")
        if not references or len(set(references)) != len(references):
            raise ValueError("retired generation refs must be non-empty and unique")
        with self._lock:
            states = tuple(
                self._state_for_generation_ref_locked(
                    reference,
                    allow_retired=True,
                )
                for reference in references
            )
            if any(not state.retired for state in states):
                raise RuntimeError("signal generation was not withdrawn")
            for state in states:
                self._drop_state_locked(state)
            self._membership_changed = True
        return self._cleanup_retired_states(states)

    def attach(
        self,
        node,
        slot,
        *,
        event_source: SignalEventSource | None = None,
    ) -> None:
        if slot is None:
            raise ValueError("a monitor slot is required")
        if not callable(getattr(slot, "freeze_live_outputs", None)):
            raise TypeError(
                "live slot must expose application-owned freeze_live_outputs()"
            )
        if event_source is not None and not isinstance(
            event_source,
            SignalEventSource,
        ):
            raise TypeError("event_source must implement SignalEventSource")
        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.terminal
                or state.kind != "producer"
                or state.node is not node
                or state.output_names != output_names
                or dict(state.bare_names) != dict(bare_names)
                or state.slot is not None
            ):
                raise RuntimeError(
                    "live output requires the producer's reserved generation"
                )
            state.slot = slot
            state.event_source = event_source
            self._membership_changed = True

    def mark_changed(self, node, slot) -> None:
        """Dirty only the exact live slot that emitted this wake."""

        if slot is None:
            raise ValueError("changed live slot must not be None")
        owner_id = _node_instance_id(node)
        with self._lock:
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.terminal
                or state.node is not node
                or state.slot is not slot
            ):
                return
            self._dirty.add(owner_id)
            output_names = frozenset(state.output_names)
            wake = (
                self._request_owner_wake
                if any(
                    candidate.source_name in output_names
                    for candidate in self._states.values()
                    if (
                        not candidate.retired
                        and candidate.kind in {"processor", "continuous"}
                    )
                )
                else None
            )
        if wake is not None:
            wake()

    def latest_publication(self, signal_name: str) -> SignalPublication | None:
        name = canonical_text(signal_name, "signal name")
        with self._lock:
            state = self._state_for_signal_locked(name)
            return None if state is None else state.publication

    def direct_parent_publications(
        self,
        publication: SignalPublication,
    ) -> tuple[SignalPublication, ...]:
        """Resolve only one issued event's exact process-local parent payloads."""

        with self._lock:
            return self._resolved_direct_parents_locked(publication)

    def publication_owner(self, publication: SignalPublication) -> object | None:
        """Return the exact active generation owner of one issued publication.

        Composition occasionally needs generation-static leaf facts that live
        on the admitted node (for example Calibration geometry).  Resolving
        those facts by scanning a current UI row with the same textual owner id
        can splice an older retained front into a restarted node.  The plane is
        already the sole generation owner, so it alone may resolve this narrow
        process-local reference.  No presentation state is stored here.
        """

        with self._lock:
            self._require_issued_publication_locked(publication)
            state = self._states.get(publication.event_ref.stream_id.value)
            if (
                state is None
                or state.retired
                or state.generation != publication.event_ref.generation
            ):
                return None
            return state.node

    def attach_latest_only_processor(
        self,
        node: object,
        *,
        source_name: str,
        initial_publication: SignalPublication,
    ) -> None:
        source_name = canonical_text(source_name, "processor source name")
        if not isinstance(initial_publication, SignalPublication):
            raise TypeError("Processor requires an exact SignalPublication")
        source = initial_publication.value(source_name)
        if source is None:
            raise ValueError("Processor publication has no selected signal")
        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            self._require_issued_publication_locked(initial_publication)
            source_state = self._state_for_signal_locked(source_name)
            if (
                source_state is None
                or source_state.publication is not initial_publication
            ):
                raise RuntimeError(
                    "Processor source is not the exact current publication"
                )
            if source_state.terminal:
                raise RuntimeError("Processor source generation is not live")
            state = self._install_state_locked(
                owner_id=owner_id,
                kind="processor",
                output_names=output_names,
                bare_names=bare_names,
                node=node,
                source_name=source_name,
            )
        try:
            self._lane.attach_processor(
                node,
                source_name,
                initial_publication,
            )
        except BaseException:
            with self._lock:
                if self._states.get(owner_id) is state:
                    self._drop_state_locked(state)
                    self._membership_changed = True
            raise

    def cancel_latest_only_processor(self, node: object) -> bool:
        idle = self._lane.cancel_processor(node)
        self.withdraw_processor(node)
        return idle

    def bind_processor_event_source(
        self,
        node: object,
        event_source: SignalEventSource,
    ) -> None:
        if not isinstance(event_source, SignalEventSource):
            raise TypeError("event_source must implement SignalEventSource")
        owner_id = _node_instance_id(node)
        with self._lock:
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.kind != "processor"
                or state.node is not node
            ):
                raise RuntimeError("Processor generation is no longer active")
            if state.publication is not None:
                raise RuntimeError(
                    "Processor event source must be bound before first publication"
                )
            if state.event_source is not None:
                raise RuntimeError("Processor event source is already bound")
            state.event_source = event_source

    def withdraw_processor(self, node: object) -> None:
        self._withdraw_owner(_node_instance_id(node))

    def _require_issued_publication_locked(
        self,
        publication: SignalPublication,
    ) -> None:
        if not isinstance(publication, SignalPublication):
            raise TypeError("signal parent must be SignalPublication")
        if publication._issuer is not self._publication_issuer:
            raise ValueError("signal publication was not issued by this data plane")

    def _resolved_direct_parents_locked(
        self,
        publication: SignalPublication,
    ) -> tuple[SignalPublication, ...]:
        """Resolve private parent payloads without weakening public lineage."""

        self._require_issued_publication_locked(publication)
        try:
            parents = self._publication_parents[publication]
        except KeyError as error:
            raise RuntimeError(
                "signal publication parent payloads are no longer retained"
            ) from error
        if (
            tuple(parent.event_ref for parent in parents)
            != publication.direct_parent_refs
        ):
            raise RuntimeError("signal publication parent refs are inconsistent")
        return parents

    def _require_route_parent_locked(
        self,
        state: _GenerationState,
        publication: SignalPublication,
    ) -> SignalValue:
        self._require_issued_publication_locked(publication)
        source_name = state.source_name
        if source_name is None:
            raise RuntimeError("derived generation has no frozen source")
        if (
            publication.event_ref.stream_id.value != state.source_owner_id
            or publication.event_ref.generation != state.source_generation
        ):
            raise ValueError("signal parent belongs to another source generation")
        source = publication.value(source_name)
        if source is None:
            raise ValueError("signal parent lacks the frozen route source")
        return source

    def _require_route_parents_locked(
        self,
        state: _GenerationState,
        publications: tuple[SignalPublication, ...],
        *,
        exact_bound: bool = False,
    ) -> tuple[SignalValue, ...]:
        parents = tuple(publications)
        if not parents or any(
            not isinstance(parent, SignalPublication) for parent in parents
        ):
            raise TypeError("derived result requires exact parent publications")
        if len({id(parent) for parent in parents}) != len(parents):
            raise ValueError("derived result parent publications must be unique")
        sequences = tuple(parent.event_ref.sequence for parent in parents)
        if sequences != tuple(sorted(sequences)):
            raise ValueError("derived result parents must follow source event order")
        values = tuple(
            self._require_route_parent_locked(state, parent) for parent in parents
        )
        if exact_bound and (
            len(parents) != len(state.bound_parents)
            or any(
                parent is not bound
                for parent, bound in zip(parents, state.bound_parents, strict=True)
            )
        ):
            raise ValueError("event result publications differ from their bound parents")
        return values

    def _validate_generation_values_locked(
        self,
        state: _GenerationState,
        values: Mapping[str, SignalValue],
        *,
        terminal: bool,
    ) -> None:
        names = tuple(values)
        if not names or not set(names).issubset(state.output_names):
            raise ValueError(
                "signal publication is outside its declared output vocabulary"
            )
        canonical_names = tuple(
            name for name in state.output_names if name in values
        )
        if not terminal:
            if state.published_names is None:
                state.published_names = canonical_names
            elif state.published_names != canonical_names:
                raise ValueError(
                    "live sibling bundle changed inside one generation"
                )
        schemas = {
            name: values[name].snapshot.block.schema
            for name in canonical_names
        }
        if any(not isinstance(schema, DatasetSchema) for schema in schemas.values()):
            raise TypeError("signal publication block must own a DatasetSchema")
        prior = {} if state.published_schemas is None else dict(
            state.published_schemas
        )
        for name, schema in schemas.items():
            if name in prior and prior[name] != schema:
                raise ValueError(
                    "signal publication schema changed inside one generation"
                )
            prior[name] = schema
        state.published_schemas = MappingProxyType(prior)

    def _publish_locked(
        self,
        state: _GenerationState,
        values: Mapping[str, SignalValue],
        *,
        parents: tuple[SignalPublication, ...] = (),
        terminal: bool = False,
    ) -> SignalPublication:
        if state.retired or self._states.get(state.owner_id) is not state:
            raise RuntimeError("signal generation is no longer active")
        if state.terminal:
            raise RuntimeError("signal generation has already published terminal")
        if len({id(parent) for parent in parents}) != len(parents):
            raise ValueError("signal publication parents must be unique")
        for parent in parents:
            self._require_issued_publication_locked(parent)
        frozen = MappingProxyType(
            {
                name: values[name]
                for name in state.output_names
                if name in values
            }
        )
        self._validate_generation_values_locked(
            state,
            frozen,
            terminal=terminal,
        )
        publication = SignalPublication(
            event_ref=EventRef(
                StreamId(state.owner_id),
                state.generation,
                state.next_sequence,
            ),
            signals=frozen,
            _issuer=self._publication_issuer,
            direct_parent_refs=tuple(parent.event_ref for parent in parents),
        )
        self._publication_parents[publication] = parents
        state.next_sequence += 1
        state.publication = publication
        state.failure = None
        state.terminal = terminal
        if terminal:
            self._dirty.discard(state.owner_id)
        self._membership_changed = True
        return publication

    def publish_final(
        self,
        node,
        projected: Mapping[str, FinalDatasetOutput],
    ) -> Mapping[str, SignalValue]:
        if not isinstance(projected, Mapping):
            raise TypeError("projected FINAL signals must be a mapping")
        owner_id = _node_instance_id(node)
        output_names, bare_names = self._node_route_names(node)
        declared = _declared_outputs(
            _require_signal_producer(node).dataset_output_declarations
        )
        if not projected or not set(projected).issubset(declared):
            raise ValueError(
                "FINAL publication must be a non-empty declared output subset"
            )
        values: dict[str, SignalValue] = {}
        by_bare = {bare: qualified for qualified, bare in bare_names.items()}
        for bare in declared:
            if bare not in projected:
                continue
            output = projected[bare]
            if not isinstance(output, FinalDatasetOutput):
                raise TypeError("FINAL values must be FinalDatasetOutput")
            _require_published_declaration(bare, output, declared)
            qualified = by_bare[bare]
            values[qualified] = SignalValue(
                name=qualified,
                snapshot=output.snapshot,
                coverage=None,
                transient=False,
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.node is not node
                or state.kind != "producer"
                or state.output_names != output_names
                or dict(state.bare_names) != dict(bare_names)
            ):
                raise RuntimeError("FINAL owner differs from its active generation")
            publication = self._publish_locked(
                state,
                values,
                terminal=True,
            )
        return publication.signals

    def publish_processor(
        self,
        node,
        outputs: Mapping[str, LiveDatasetOutput],
        *,
        source_publication: SignalPublication,
    ) -> Mapping[str, SignalValue]:
        if not isinstance(outputs, Mapping):
            raise TypeError("processor outputs must be a mapping")
        if not isinstance(source_publication, SignalPublication):
            raise TypeError("Processor publication requires its exact parent")
        owner_id = _node_instance_id(node)
        declared = _declared_outputs(
            _require_signal_producer(node).dataset_output_declarations
        )
        if set(outputs) != set(declared):
            raise ValueError(
                "Processor publication must cover its complete frozen output vocabulary"
            )
        with self._lock:
            state = self._states.get(owner_id)
            if (
                state is None
                or state.retired
                or state.kind != "processor"
                or state.node is not node
            ):
                raise RuntimeError("Processor generation is no longer active")
            self._require_route_parent_locked(state, source_publication)
            if source_publication.event_ref.sequence <= state.last_parent_sequence:
                raise RuntimeError("Processor result belongs to an obsolete parent")
            source_name = state.source_name
            bare_names = dict(state.bare_names)
        assert source_name is not None
        source = source_publication.value(source_name)
        if source is None:
            raise ValueError("Processor parent lacks its selected signal")
        by_bare = {bare: qualified for qualified, bare in bare_names.items()}
        values: dict[str, SignalValue] = {}
        for bare in declared:
            output = outputs[bare]
            if not isinstance(output, LiveDatasetOutput):
                raise TypeError("processor outputs must be LiveDatasetOutput")
            _require_published_declaration(bare, output, declared)
            qualified = by_bare[bare]
            values[qualified] = SignalValue(
                name=qualified,
                snapshot=output.snapshot,
                coverage=output.coverage,
                transient=True,
            )
        with self._lock:
            if self._states.get(owner_id) is not state or state.retired:
                raise RuntimeError("Processor generation retired during publication")
            self._require_route_parent_locked(state, source_publication)
            if source_publication.event_ref.sequence <= state.last_parent_sequence:
                raise RuntimeError("Processor result belongs to an obsolete parent")
            publication = self._publish_locked(
                state,
                values,
                parents=(source_publication,),
            )
            state.last_parent_sequence = source_publication.event_ref.sequence
        return publication.signals

    def _event_binding_locked(
        self,
        signal_name: str,
    ) -> tuple[SignalEventSource, str]:
        state = self._state_for_signal_locked(signal_name)
        if state is None:
            raise LookupError(f"signal {signal_name!r} is not active")
        source = state.event_source
        if source is None:
            raise TypeError(f"signal {signal_name!r} has no event-source capability")
        output_name = state.event_output_name
        if output_name is None:
            output_name = state.bare_names[signal_name]
        return source, output_name

    def signal_event_binding(
        self,
        signal_name: str,
        *,
        expected_generation: StreamGenerationId | None = None,
    ) -> tuple[
        StreamGenerationId,
        SignalEventSource,
        str,
    ]:
        """Freeze one active producer-owned event capability.

        A passive externally-triggered producer can be armed and association-
        capable before its first event exists.  A derived generation may expose
        an already-projected capability supplied by its composition owner; the
        plane never stores or reconstructs the projection's selector semantics.
        """

        name = canonical_text(signal_name, "signal event route name")
        if expected_generation is not None and not isinstance(
            expected_generation,
            StreamGenerationId,
        ):
            raise TypeError("expected_generation must be StreamGenerationId")
        with self._lock:
            selected = self._state_for_signal_locked(name)
            if selected is None or selected.retired:
                raise LookupError(f"signal {name!r} is not active")
            if (
                expected_generation is not None
                and selected.generation != expected_generation
            ):
                raise RuntimeError("signal generation changed before event binding")
            source, output_name = self._event_binding_locked(name)
            return (
                selected.generation,
                source,
                output_name,
            )

    def has_event_association(self, signal_name: str) -> bool:
        try:
            _generation, source, _output_name = self.signal_event_binding(signal_name)
        except (KeyError, LookupError, RuntimeError, TypeError, ValueError):
            return False
        return isinstance(source, SignalEventAssociationSource)

    def bind_continuous_derived(
        self,
        owner_id: str,
        *,
        source_name: str,
        expected_source_generation: StreamGenerationId,
        output_names,
        event_source: SignalEventSource | None = None,
        event_output_name: str | None = None,
    ) -> StreamGenerationId:
        """Bind one derived sibling bundle to its direct source generation.

        ``event_source`` is an optional, already-projected capability owned by
        the producer/frontend composition seam.  Selector transforms and other
        physical configuration never enter this routing state.
        """

        identity = canonical_text(owner_id, "derived route owner_id")
        source_name = canonical_text(source_name, "derived route source name")
        if not isinstance(expected_source_generation, StreamGenerationId):
            raise TypeError("expected_source_generation must be StreamGenerationId")
        names = tuple(
            canonical_text(name, "derived route output name")
            for name in output_names
        )
        if event_source is not None and len(names) != 1:
            raise ValueError("one event capability can bind exactly one derived output")
        if event_source is not None:
            if not isinstance(event_source, SignalEventSource):
                raise TypeError("event_source must implement SignalEventSource")
            if event_output_name is None:
                raise ValueError("event capability requires its output name")
            event_output_name = canonical_text(
                event_output_name,
                "derived event output name",
            )
            event_schema = event_source.value_schema(event_output_name)
            if not isinstance(event_schema, ValueSchema):
                raise TypeError("signal event source must return ValueSchema")
        elif event_output_name is not None:
            raise ValueError("event output name requires an event_source")
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            source_state = self._state_for_signal_locked(source_name)
            if source_state is None:
                raise RuntimeError("derived route source generation is not active")
            if source_state.generation != expected_source_generation:
                raise RuntimeError("derived route source generation changed")
            existing = self._states.get(identity)
            if existing is not None and not existing.retired:
                same = (
                    existing.kind == "continuous"
                    and existing.source_name == source_name
                    and existing.source_generation == expected_source_generation
                    and existing.output_names == names
                    and existing.event_source is event_source
                    and existing.event_output_name == event_output_name
                )
                if same:
                    return existing.generation
                raise RuntimeError(
                    "derived generation changed; withdraw it before rebinding"
                )
            state = self._install_state_locked(
                owner_id=identity,
                kind="continuous",
                output_names=names,
                bare_names={name: name for name in names},
                source_name=source_name,
                event_source=event_source,
                event_output_name=event_output_name,
            )
            return state.generation

    @staticmethod
    def _route_owned_snapshot(
        state: _GenerationState,
        output_name: str,
        snapshot: OwnedSnapshot,
    ) -> OwnedSnapshot:
        """Bind one derived value to the plane-owned route generation.

        Materializers own values and schemas, but not live route identity.
        Assigning the Dataset ref here prevents frontend transforms, Fits, or
        payload hashes from becoming a parallel generation authority.  Values
        already crossed the immutable ownership boundary, so rebuilding the
        small DataBlock header retains their bytes-backed arrays without a
        scientific-data copy.
        """

        block_id = BlockId(f"signal/{state.owner_id}/{output_name}")
        ref = DatasetRevisionRef(
            block_id,
            state.generation,
            snapshot.block.schema.fingerprint,
            snapshot.ref.revision,
        )
        return OwnedSnapshot(
            ref,
            DataBlock(
                block_id,
                snapshot.block.revision,
                snapshot.block.values,
                snapshot.block.validity,
                snapshot.block.schema,
            ),
        )

    @staticmethod
    def _derived_values(
        state: _GenerationState,
        source_publication: SignalPublication,
        values: Mapping[str, DerivedSignalOutput],
        *,
        transient: bool,
    ) -> Mapping[str, SignalValue]:
        source_name = state.source_name
        if source_name is None:
            raise RuntimeError("derived generation has no source")
        source = source_publication.value(source_name)
        if source is None:
            raise ValueError("derived parent lacks its selected signal")
        if set(values) != set(state.output_names):
            raise ValueError(
                "derived publication differs from its frozen sibling vocabulary"
            )
        result = {}
        for name in state.output_names:
            value = values[name]
            if not isinstance(value, DerivedSignalOutput):
                raise TypeError("derived values must contain DerivedSignalOutput")
            if state.event_source is not None:
                assert state.event_output_name is not None
                expected_schema = state.event_source.value_schema(
                    state.event_output_name
                )
                if not isinstance(expected_schema, ValueSchema):
                    raise TypeError("signal event source must return ValueSchema")
                if value.snapshot.block.schema.cell_schema != expected_schema:
                    raise ValueError(
                        "derived output schema differs from its event capability"
                    )
            result[name] = SignalValue(
                name=name,
                snapshot=SignalDataPlane._route_owned_snapshot(
                    state,
                    name,
                    value.snapshot,
                ),
                coverage=(
                    source.coverage if value.preserve_source_coverage else None
                ),
                transient=transient,
            )
        return MappingProxyType(result)

    def publish_continuous_derived(
        self,
        owner_id: str,
        generation: StreamGenerationId,
        source_publications: tuple[SignalPublication, ...],
        values: Mapping[str, DerivedSignalOutput],
    ) -> bool:
        identity = canonical_text(owner_id, "derived route owner_id")
        if not isinstance(generation, StreamGenerationId):
            raise TypeError("derived generation must be StreamGenerationId")
        parents = tuple(source_publications)
        with self._lock:
            state = self._states.get(identity)
            if (
                state is None
                or state.retired
                or state.generation != generation
                or state.kind != "continuous"
            ):
                return False
            self._require_route_parents_locked(state, parents)
            latest_sequence = parents[-1].event_ref.sequence
            if latest_sequence <= state.last_parent_sequence:
                return False
        frozen = self._derived_values(
            state,
            parents[-1],
            values,
            transient=True,
        )
        with self._lock:
            if (
                self._states.get(identity) is not state
                or state.retired
                or state.generation != generation
            ):
                return False
            self._require_route_parents_locked(state, parents)
            latest_sequence = parents[-1].event_ref.sequence
            if latest_sequence <= state.last_parent_sequence:
                return False
            self._publish_locked(
                state,
                frozen,
                parents=parents,
            )
            state.last_parent_sequence = latest_sequence
            return True

    def fail_continuous_derived(
        self,
        owner_id: str,
        generation: StreamGenerationId,
        source_publications: tuple[SignalPublication, ...],
        error: Exception,
    ) -> bool:
        identity = canonical_text(owner_id, "derived route owner_id")
        parents = tuple(source_publications)
        if not isinstance(error, Exception):
            raise TypeError("derived failure must be an Exception")
        with self._lock:
            state = self._states.get(identity)
            if (
                state is None
                or state.retired
                or state.generation != generation
                or state.kind != "continuous"
            ):
                return False
            self._require_route_parents_locked(state, parents)
            latest_sequence = parents[-1].event_ref.sequence
            if latest_sequence <= state.last_parent_sequence:
                return False
            state.last_parent_sequence = latest_sequence
            state.failure = f"{type(error).__name__}: {error}"
            self._membership_changed = True
            return True

    def continuous_needs_publication(
        self,
        owner_id: str,
        generation: StreamGenerationId,
        source_publications: tuple[SignalPublication, ...],
    ) -> bool:
        """Whether one active route has not published this exact parent yet."""

        identity = canonical_text(owner_id, "derived route owner_id")
        parents = tuple(source_publications)
        with self._lock:
            state = self._states.get(identity)
            if (
                state is None
                or state.retired
                or state.kind != "continuous"
                or state.generation != generation
            ):
                return False
            self._require_route_parents_locked(state, parents)
            return parents[-1].event_ref.sequence > state.last_parent_sequence

    def bind_event_derived(
        self,
        owner_id: str,
        *,
        source_name: str,
        source_publications: tuple[SignalPublication, ...],
        output_names,
    ) -> StreamGenerationId:
        identity = canonical_text(owner_id, "event result owner_id")
        source_name = canonical_text(source_name, "event result source name")
        names = tuple(
            canonical_text(name, "event result output name")
            for name in output_names
        )
        parents = tuple(source_publications)
        if not parents or any(
            not isinstance(parent, SignalPublication) for parent in parents
        ):
            raise TypeError("event result binding requires exact parents")
        if any(parent.value(source_name) is None for parent in parents):
            raise ValueError("event result parent lacks its selected signal")
        self._withdraw_owner(identity)
        with self._lock:
            if self._closed:
                raise RuntimeError("signal data plane is closed")
            for parent in parents:
                self._require_issued_publication_locked(parent)
            primary = parents[-1]
            source_state = self._state_for_signal_locked(source_name)
            if (
                source_state is None
                or source_state.publication is None
                or source_state.owner_id
                != primary.event_ref.stream_id.value
                or source_state.generation != primary.event_ref.generation
            ):
                raise RuntimeError("event result source has no exact publication")
            state = self._install_state_locked(
                owner_id=identity,
                kind="event",
                output_names=names,
                bare_names={name: name for name in names},
                source_name=source_name,
            )
            self._require_route_parents_locked(state, parents)
            state.bound_parents = parents
            return state.generation

    def publish_event_derived(
        self,
        owner_id: str,
        generation: StreamGenerationId,
        source_publications: tuple[SignalPublication, ...],
        values: Mapping[str, DerivedSignalOutput],
    ) -> Mapping[str, SignalValue] | None:
        identity = canonical_text(owner_id, "event result owner_id")
        if not isinstance(generation, StreamGenerationId):
            raise TypeError("event result generation must be StreamGenerationId")
        parents = tuple(source_publications)
        with self._lock:
            state = self._states.get(identity)
            if (
                state is None
                or state.retired
                or state.generation != generation
                or state.kind != "event"
            ):
                return None
            source_name = state.source_name
            self._require_route_parents_locked(
                state,
                parents,
                exact_bound=True,
            )
        if source_name is None:
            raise RuntimeError("event result generation lost its source")
        frozen = self._derived_values(
            state,
            parents[-1],
            values,
            transient=True,
        )
        with self._lock:
            if (
                self._states.get(identity) is not state
                or state.retired
                or state.generation != generation
            ):
                return None
            self._require_route_parents_locked(
                state,
                parents,
                exact_bound=True,
            )
            publication = self._publish_locked(
                state,
                frozen,
                parents=parents,
                terminal=True,
            )
        return publication.signals

    def withdraw_derived(self, owner_id: str) -> None:
        self._withdraw_owner(
            canonical_text(owner_id, "derived signal owner_id")
        )

    def _retirement_closure_locked(
        self,
        root_owner_id: str,
    ) -> tuple[_GenerationState, ...]:
        root = self._states.get(root_owner_id)
        if root is None or root.retired:
            return ()
        selected = {(root.owner_id, root.generation)}
        changed = True
        while changed:
            changed = False
            for state in self._states.values():
                reference = self._generation_ref(state)
                source_ref = (
                    None
                    if state.source_owner_id is None
                    else (state.source_owner_id, state.source_generation)
                )
                if state.retired or reference in selected:
                    continue
                if source_ref in selected:
                    selected.add(reference)
                    changed = True
        states = tuple(
            state
            for state in self._states.values()
            if self._generation_ref(state) in selected
        )
        for state in states:
            self._drop_state_locked(state)
        self._membership_changed = True
        return states

    def _withdraw_owner(self, owner_id: str) -> frozenset[str]:
        with self._lock:
            states = self._retirement_closure_locked(owner_id)
        retired_names = frozenset(
            name for state in states for name in state.output_names
        )
        errors = self._cleanup_retired_states(states)
        if errors:
            raise BaseExceptionGroup(
                "signal generation cleanup failed",
                list(errors),
            )
        return retired_names

    def _cleanup_retired_states(
        self,
        states: tuple[_GenerationState, ...],
    ) -> tuple[BaseException, ...]:
        slots = []
        for state in states:
            if state.kind == "processor" and state.node is not None:
                # Routing retirement and execution retirement are distinct.
                # Keep a cancelled entry lane-owned until its prepare/work
                # Future completes so the node receives exactly one terminal
                # acknowledgement from ``_cancelled_processor``.
                self._lane.cancel_processor(state.node)
            if state.slot is not None:
                slots.append(state.slot)
        errors = []
        seen_slots = set()
        for slot in slots:
            if id(slot) in seen_slots:
                continue
            seen_slots.add(id(slot))
            try:
                slot.close()
            except BaseException as error:
                errors.append(error)
        return tuple(errors)

    def retire(self, node: object) -> frozenset[str]:
        owner_id = _node_instance_id(node)
        with self._lock:
            state = self._states.get(owner_id)
            if state is not None and state.node is not node:
                raise RuntimeError("signal owner id belongs to another generation")
        return self._withdraw_owner(owner_id)

    def detach_live(self, node) -> None:
        owner_id = _node_instance_id(node)
        retained_slot = None
        retired_states = ()
        with self._lock:
            state = self._states.get(owner_id)
            if state is None:
                return
            if state.node is not node:
                raise RuntimeError("signal owner id belongs to another generation")
            retained_final = bool(
                state.publication is not None
                and all(
                    not value.transient
                    for value in state.publication.signals.values()
                )
            )
            if retained_final:
                retained_slot = state.slot
                state.slot = None
                self._dirty.discard(owner_id)
            else:
                # Withdraw this exact generation and its descendants before
                # any slow slot close/join.  A replacement may then reserve
                # safely; cleanup below carries the retired state objects, not
                # the reusable textual owner id.
                retired_states = self._retirement_closure_locked(owner_id)
        if retained_slot is not None:
            retained_slot.close()
        errors = self._cleanup_retired_states(retired_states)
        if errors:
            raise BaseExceptionGroup(
                "signal generation cleanup failed",
                list(errors),
            )

    def _freeze_one(
        self,
        state: _GenerationState,
    ) -> tuple[Mapping[str, SignalValue], str | None]:
        node = state.node
        slot = state.slot
        if node is None or slot is None:
            raise RuntimeError("live generation lost its producer slot")
        outputs = slot.freeze_live_outputs()
        if not isinstance(outputs, Mapping) or not outputs:
            raise ValueError("live output owner must return a non-empty mapping")
        declared = _declared_outputs(
            _require_signal_producer(node).dataset_output_declarations
        )
        if not set(outputs).issubset(declared):
            raise ValueError(
                "live publication contains an undeclared output"
            )
        by_bare = {
            bare: qualified
            for qualified, bare in state.bare_names.items()
        }
        frozen: dict[str, SignalValue] = {}
        for bare in declared:
            if bare not in outputs:
                continue
            output = outputs[bare]
            if not isinstance(output, LiveDatasetOutput):
                raise TypeError("live output values must be LiveDatasetOutput")
            _require_published_declaration(bare, output, declared)
            qualified = by_bare[bare]
            frozen[qualified] = SignalValue(
                name=qualified,
                snapshot=output.snapshot,
                coverage=output.coverage,
                transient=True,
            )
        return (
            MappingProxyType(frozen),
            getattr(slot, "notification_failure", None),
        )

    def _publication_roots(
        self,
        publication: SignalPublication,
    ) -> frozenset[EventRef]:
        pending = [publication]
        roots = set()
        seen = set()
        while pending:
            current = pending.pop()
            if current.event_ref in seen:
                continue
            seen.add(current.event_ref)
            parents = self._resolved_direct_parents_locked(current)
            if parents:
                pending.extend(parents)
            else:
                roots.add(current.event_ref)
        return frozenset(roots)

    def _collect_publication_ancestry(
        self,
        publication: SignalPublication,
    ) -> Mapping[str, SignalPublication] | None:
        pending = [publication]
        by_name: dict[str, SignalPublication] = {}
        seen = set()
        while pending:
            current = pending.pop()
            if current.event_ref in seen:
                continue
            seen.add(current.event_ref)
            for name in current.signals:
                previous = by_name.get(name)
                if (
                    previous is not None
                    and previous.event_ref != current.event_ref
                ):
                    return None
                by_name[name] = current
            pending.extend(self._resolved_direct_parents_locked(current))
        return by_name

    @staticmethod
    def _name_is_ancestor(
        ancestor: str,
        descendant: str,
        source_by_output: Mapping[str, str],
    ) -> bool:
        seen = set()
        current = descendant
        while current in source_by_output and current not in seen:
            seen.add(current)
            current = source_by_output[current]
            if current == ancestor:
                return True
        return False

    def _build_front_locked(self) -> SignalFront:
        latest: dict[str, SignalPublication] = {}
        failures = {}
        active_names = set()
        event_names = set()
        adjacency: dict[str, set[str]] = {}
        source_by_output: dict[str, str] = {}

        for state in self._states.values():
            if state.retired:
                continue
            if state.failure is not None:
                failures[state.owner_id] = state.failure
            publication = state.publication
            if state.terminal and publication is not None:
                state_active_names = set(publication.signals)
            else:
                state_active_names = set(state.output_names)
            active_names.update(state_active_names)
            if state.kind == "event":
                event_names.update(state_active_names)
            if publication is not None:
                for name in publication.signals:
                    if name in latest:
                        raise RuntimeError(f"signal {name!r} has two publications")
                    latest[name] = publication
            siblings = () if publication is None else tuple(publication.signals)
            for name in siblings:
                adjacency.setdefault(name, set()).update(
                    candidate for candidate in siblings if candidate != name
                )
            if (
                state.kind in {"processor", "continuous"}
                and state.source_name is not None
            ):
                for output_name in state.output_names:
                    source_by_output[output_name] = state.source_name
                    adjacency.setdefault(output_name, set()).add(state.source_name)
                    adjacency.setdefault(state.source_name, set()).add(output_name)

        selected = dict(latest)
        requested = self._front_signals.intersection(active_names).difference(
            event_names
        )
        pending = set(requested)
        components = []
        while pending:
            seed = pending.pop()
            component = {seed}
            stack = [seed]
            while stack:
                current = stack.pop()
                for neighbour in adjacency.get(current, ()):
                    if neighbour in active_names and neighbour not in component:
                        component.add(neighbour)
                        stack.append(neighbour)
            pending.difference_update(component)
            components.append(component)

        previous_publications = self._front.publication_by_signal
        for component in components:
            requested_component = requested.intersection(component)
            leaves = tuple(
                name
                for name in requested_component
                if not any(
                    other != name
                    and self._name_is_ancestor(name, other, source_by_output)
                    for other in requested_component
                )
            )
            leaf_publications = tuple(
                latest.get(name)
                for name in leaves
            )
            coherent = bool(leaves) and all(
                publication is not None
                for publication in leaf_publications
            )
            ancestry: dict[str, SignalPublication] = {}
            if coherent:
                root_sets = {
                    self._publication_roots(publication)
                    for publication in leaf_publications
                    if publication is not None
                }
                coherent = len(root_sets) == 1
            if coherent:
                for publication in leaf_publications:
                    assert publication is not None
                    current = self._collect_publication_ancestry(publication)
                    if current is None:
                        coherent = False
                        break
                    for name, candidate in current.items():
                        previous = ancestry.get(name)
                        if (
                            previous is not None
                            and previous.event_ref != candidate.event_ref
                        ):
                            coherent = False
                            break
                        ancestry[name] = candidate
                    if not coherent:
                        break
            if coherent and any(
                name not in ancestry for name in requested_component
            ):
                coherent = False

            if coherent:
                for name, publication in ancestry.items():
                    if name in component and name in active_names:
                        selected[name] = publication
            else:
                for name in component:
                    previous = previous_publications.get(name)
                    current_state = self._state_for_signal_locked(name)
                    if (
                        previous is None
                        or name not in active_names
                        or current_state is None
                        or current_state.owner_id
                        != previous.event_ref.stream_id.value
                        or current_state.generation
                        != previous.event_ref.generation
                    ):
                        selected.pop(name, None)
                    else:
                        selected[name] = previous

        signals = {}
        publications = {}
        for name, publication in selected.items():
            if name not in active_names:
                continue
            value = publication.value(name)
            if value is None:
                raise RuntimeError("front publication lost one of its signals")
            signals[name] = value
            publications[name] = publication
        groups = {}
        for component in components:
            members = frozenset(
                name
                for name in requested.intersection(component)
                if name in signals
            )
            for name in members:
                groups[name] = members
        return SignalFront(
            signals,
            failures,
            publications,
            groups,
        )

    def freeze(self) -> SignalFront:
        """Advance changed generations and return one immutable coherent front."""

        self._lane.drain_processors()
        with self._lock:
            if not self._dirty and not self._membership_changed:
                return self._front
            dirty = tuple(self._dirty)
            self._dirty.clear()
            states = {
                owner_id: self._states.get(owner_id)
                for owner_id in dirty
            }
            self._membership_changed = False

        for owner_id, state in states.items():
            if (
                state is None
                or state.retired
                or state.terminal
                or state.slot is None
            ):
                continue
            try:
                values, warning = self._freeze_one(state)
            except Exception as error:
                with self._lock:
                    if (
                        self._states.get(owner_id) is state
                        and not state.retired
                        and not state.terminal
                    ):
                        state.failure = f"{type(error).__name__}: {error}"
                        self._membership_changed = True
                continue
            try:
                with self._lock:
                    if (
                        self._states.get(owner_id) is state
                        and not state.retired
                        and not state.terminal
                    ):
                        self._publish_locked(state, values)
                        state.failure = warning
            except Exception as error:
                with self._lock:
                    if (
                        self._states.get(owner_id) is state
                        and not state.retired
                        and not state.terminal
                    ):
                        state.failure = f"{type(error).__name__}: {error}"
                        self._membership_changed = True

        with self._lock:
            latest = {
                name: publication
                for state in self._states.values()
                if not state.retired and state.publication is not None
                for name, publication in (
                    (name, state.publication)
                    for name in state.publication.signals
                )
            }
        self._lane.route(latest)
        with self._lock:
            front = self._build_front_locked()
            self._front = front
            self._membership_changed = False
            return front

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            states = tuple(self._states.values())
            self._states.clear()
            self._dirty.clear()
            self._front_signals = frozenset()
            self._request_owner_wake = None
            self._owner_wake_token = None
            self._front = SignalFront({}, {})
            self._publication_parents.clear()
        self._lane.close()
        errors = []
        seen_slots = set()
        for state in states:
            slot = state.slot
            if slot is None or id(slot) in seen_slots:
                continue
            seen_slots.add(id(slot))
            try:
                slot.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("signal data plane close failed", errors)

    def __len__(self) -> int:
        with self._lock:
            return sum(
                state.slot is not None and not state.retired
                for state in self._states.values()
            )
