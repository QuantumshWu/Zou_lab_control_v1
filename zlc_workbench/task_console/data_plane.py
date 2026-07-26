"""MONITOR seam: change-driven fronts with local causal coherence.

Monitor seam of the composition root (``app.py``).  A run node
publishes into a ``LiveDatasetSlot``; this module freezes only slots that
reported a new revision.  Each slot is one producer transaction.  Combining
their latest immutable fronts into one presentation cycle does *not* assert
that independent producers observed the same physical shot.

Freeze-latest, not a bus.  Each changed slot materialises its own atomic
transaction exactly once; unchanged slots reuse their immutable fronts.  Independent
producers still advance independently.  Within one explicit source -> Processor
component, however, a newer source and its active descendants replace the previous
component together.  A slow Processor therefore cannot put Camera revision N beside
its own Occupancy revision N-1.

What replaced the shot clock: a monitor tap overwrites when the display falls
behind rather than back-pressuring acquisition, and it says so per signal
through :class:`~zlc_neutral_atom.runtime.dataset.MonitorCoverage`
(written/total cells, missed events, current gap).  There is no global shot
counter to compare against, and reintroducing one would be a fiction -- signals
from different runs advance independently.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Mapping

from zlc_data import OwnedSnapshot
from zlc_frontend.figure_outputs import (
    FitParameterMetadata,
    SelectorAxisMetadata,
)
from zlc_frontend.site_map import SiteMapPresentation
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCoverage,
    MonitorCoverage,
)
from zlc_storage import canonical_text, sha256_text

__all__ = [
    "ConsoleDataFront",
    "ConsoleDataPlane",
    "ConsoleSignalValue",
]


@dataclass(frozen=True)
class ConsoleSignalValue:
    """One signal at one producer-owned immutable revision."""

    name: str
    source: str                     # presentation label of the producer
    snapshot: OwnedSnapshot
    coverage: DatasetCoverage | MonitorCoverage | None
    # Lineage, carried because only the freeze knows it: a renderer stamps what
    # it drew with the run and event it came from, and a value that lost these
    # on the way to a panel could only be presented with an invented one.
    run_id: str
    epoch_id: str                   # causation domain the run belongs to
    join_digest: str                # exact immutable source/coherence digest
    transient: bool = False         # withdrawn with its live producer
    presentation: (
        SiteMapPresentation | SelectorAxisMetadata | FitParameterMetadata | None
    ) = None

    def __post_init__(self) -> None:
        name = canonical_text(self.name, "signal name")
        source = canonical_text(self.source, "signal source")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("signal snapshot must be OwnedSnapshot")
        if self.coverage is not None and not isinstance(
            self.coverage,
            (DatasetCoverage, MonitorCoverage),
        ):
            raise TypeError("signal coverage has an unknown type")
        run_id = canonical_text(self.run_id, "signal run_id")
        epoch_id = canonical_text(self.epoch_id, "signal epoch_id")
        join_digest = sha256_text(self.join_digest, "signal join_digest")
        if not isinstance(self.transient, bool):
            raise TypeError("signal transient flag must be bool")
        if self.presentation is not None and not isinstance(
            self.presentation,
            (SiteMapPresentation, SelectorAxisMetadata, FitParameterMetadata),
        ):
            raise TypeError("Console signal presentation has an unknown type")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "epoch_id", epoch_id)
        object.__setattr__(self, "join_digest", join_digest)

    # The block is the value; these read off it rather than copying, so a panel
    # and a legend describing "the same signal" cannot describe different data.
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
        keeps the console on the same description the producer declared.
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

        This is what the display-behind advisory reads.  It is per signal, from
        the tap that actually dropped them; there is no global shot counter to
        subtract, and inventing one would mean comparing runs that advance
        independently.
        """

        if isinstance(self.coverage, MonitorCoverage):
            return self.coverage.missed_events
        return 0


@dataclass(frozen=True)
class ConsoleDataFront:
    """Immutable front: coherent derived components, independent producers."""

    signals: Mapping[str, ConsoleSignalValue]
    failures: Mapping[str, str]     # producer instance_id -> freeze failure

    def names(self) -> tuple[str, ...]:
        return tuple(self.signals)

    def value(self, name: str) -> ConsoleSignalValue | None:
        return self.signals.get(str(name))


def _declared_outputs(declarations) -> dict[str, DatasetOutputDeclaration]:
    """Return the frozen owner declarations behind Workbench presentation."""

    values = tuple(
        getattr(item, "declaration", None) for item in tuple(declarations)
    )
    if any(not isinstance(value, DatasetOutputDeclaration) for value in values):
        raise TypeError(
            "console outputs must retain DatasetOutputDeclaration values"
        )
    result = {value.name: value for value in values}
    if len(result) != len(values):
        raise ValueError("console output declarations contain duplicate names")
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
            "published output is absent from the frozen Workbench vocabulary"
        )
    if output.declaration != expected:
        raise ValueError(
            "published output contract differs from the frozen owner declaration"
        )


def _node_instance_id(node: object) -> str:
    """Return the stable producer identity required by the Workbench seam."""

    return canonical_text(
        getattr(node, "instance_id", None),
        "console producer instance_id",
    )


def _node_display_label(node: object) -> str:
    """Return presentation text without letting it participate in routing."""

    label = (
        getattr(node, "display_label", None)
        or getattr(node, "name", None)
        or type(node).__name__
    )
    return canonical_text(str(label), "console producer display label")


def _node_declared_signal_names(node: object) -> tuple[str, ...]:
    """Qualify one node's frozen owner declarations through its route owner."""

    declared = _declared_outputs(
        tuple(getattr(node, "output_declarations", ()) or ())
    )
    qualify = getattr(node, "signal_key", None)
    if not callable(qualify):
        raise TypeError("console producer exposes no signal_key()")
    return tuple(str(qualify(name)) for name in declared)


def _signal_revision_identity(value: ConsoleSignalValue) -> tuple[object, ...]:
    """Exact source identity used by the latest-only Processor lane."""

    if not isinstance(value, ConsoleSignalValue):
        raise TypeError("processor source must be ConsoleSignalValue")
    return (
        value.name,
        value.run_id,
        value.epoch_id,
        value.snapshot.ref,
        value.join_digest,
    )


def _evaluate_prepared_processor_application(
    application: object,
    source: ConsoleSignalValue,
    coverage: MonitorCoverage,
) -> object:
    """Run one domain-owned Processor operation over an admitted revision.

    The data plane knows only the common application seam: an immutable source,
    typed coverage, and its event digest go into the already-prepared domain
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
        source_event_digest=source.join_digest,
    )
    from zlc_neutral_atom.processing import require_causal_processor_evaluation

    return require_causal_processor_evaluation(
        result,
        source_ref=source.snapshot.ref,
        source_event_digest=source.join_digest,
    )


@dataclass(frozen=True, slots=True)
class _ProcessorEdgePublication:
    """One exact source -> complete Processor-output causal edge."""

    node: object
    source_name: str
    source_identity: tuple[object, ...]
    outputs: Mapping[str, ConsoleSignalValue]

    def __post_init__(self) -> None:
        source_name = canonical_text(self.source_name, "processor source name")
        source_identity = tuple(self.source_identity)
        outputs = dict(self.outputs)
        if not source_identity:
            raise ValueError("Processor edge source identity must not be empty")
        if not outputs:
            raise ValueError("Processor edge must publish outputs")
        if any(not isinstance(value, ConsoleSignalValue) for value in outputs.values()):
            raise TypeError("Processor edge outputs must be ConsoleSignalValue values")
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "source_identity", source_identity)
        object.__setattr__(self, "outputs", MappingProxyType(outputs))


@dataclass(frozen=True, slots=True)
class _ProcessorSourceComponent:
    """One root producer transaction plus its ordered causal Processor edges."""

    root_signals: Mapping[str, ConsoleSignalValue]
    edges: tuple[_ProcessorEdgePublication, ...] = ()

    def __post_init__(self) -> None:
        root = dict(self.root_signals)
        edges = tuple(self.edges)
        if not root:
            raise ValueError("Processor source component requires root signals")
        if any(not isinstance(value, ConsoleSignalValue) for value in root.values()):
            raise TypeError(
                "Processor source component must contain ConsoleSignalValue values"
            )
        accumulated = dict(root)
        for edge in edges:
            if not isinstance(edge, _ProcessorEdgePublication):
                raise TypeError("Processor source component has another edge type")
            source = accumulated.get(edge.source_name)
            if (
                source is None
                or _signal_revision_identity(source) != edge.source_identity
            ):
                raise ValueError(
                    "Processor source component edge has another source revision"
                )
            overlap = set(accumulated).intersection(edge.outputs)
            if overlap:
                raise ValueError(
                    "Processor source component edge redefines signals: "
                    + ", ".join(sorted(overlap))
                )
            accumulated.update(edge.outputs)
        object.__setattr__(self, "root_signals", MappingProxyType(root))
        object.__setattr__(self, "edges", edges)

    @property
    def signals(self) -> Mapping[str, ConsoleSignalValue]:
        values = dict(self.root_signals)
        for edge in self.edges:
            values.update(edge.outputs)
        return MappingProxyType(values)

    def extend(
        self,
        edge: _ProcessorEdgePublication,
    ) -> _ProcessorSourceComponent:
        return _ProcessorSourceComponent(self.root_signals, (*self.edges, edge))

    def through_signal(self, name: str) -> _ProcessorSourceComponent:
        selected = canonical_text(name, "processor source name")
        if selected in self.root_signals:
            return _ProcessorSourceComponent(self.root_signals)
        for index, edge in enumerate(self.edges):
            if selected in edge.outputs:
                return _ProcessorSourceComponent(
                    self.root_signals,
                    self.edges[: index + 1],
                )
        raise KeyError(selected)


@dataclass(frozen=True, slots=True)
class _ProcessorPublication:
    edge: _ProcessorEdgePublication
    source_component: _ProcessorSourceComponent

    def __post_init__(self) -> None:
        if not isinstance(self.edge, _ProcessorEdgePublication):
            raise TypeError("Processor publication has another edge type")
        if not isinstance(self.source_component, _ProcessorSourceComponent):
            raise TypeError("Processor publication has another source component type")
        self.source_component.extend(self.edge)

    def output_component(self) -> _ProcessorSourceComponent:
        return self.source_component.extend(self.edge)


@dataclass(slots=True)
class _LatestOnlyProcessorEntry:
    node: object
    source_name: str
    prepare_future: Future
    application: object | None = None
    work_future: Future | None = None
    work_source: ConsoleSignalValue | None = None
    work_source_component: _ProcessorSourceComponent | None = None
    pending_source: ConsoleSignalValue | None = None
    pending_source_component: _ProcessorSourceComponent | None = None
    last_source_identity: tuple[object, ...] | None = None
    cancel_requested: bool = False


class _LatestOnlyProcessorLane:
    """Internal shared worker lane for explicit source revisions.

    This host has no graph, scheduler policy, restart machinery, or domain
    algorithm.  It serializes already-prepared Processor operations owned by
    nodes in one TaskConsole and keeps at most the newest not-yet-run source
    revision per node.  Replacing pending work is a host scheduling decision,
    not acquisition loss: it must never rewrite the producer-owned
    ``MonitorCoverage`` carried by that revision.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="console-latest-processor",
        )
        self._entries: dict[int, _LatestOnlyProcessorEntry] = {}
        self._closed = False

    @staticmethod
    def _require_node_contract(node: object) -> None:
        for name in (
            "_prepare_processor_application",
            "_validate_processor_source",
            "_processor_application_ready",
            "_processor_work_started",
            "_accept_processor_result",
            "_accept_processor_failure",
            "_accept_processor_cancelled",
            "_request_processor_owner_wake",
        ):
            if not callable(getattr(node, name, None)):
                raise TypeError(
                    f"Processor node must implement {name}()"
                )

    def attach(
        self,
        node: object,
        source_name: str,
        initial_source: ConsoleSignalValue,
        initial_source_component: _ProcessorSourceComponent,
    ) -> None:
        if self._closed:
            raise RuntimeError("latest-only Processor lane is closed")
        self._require_node_contract(node)
        name = canonical_text(source_name, "processor source name")
        if not isinstance(initial_source, ConsoleSignalValue):
            raise TypeError("initial Processor source must be ConsoleSignalValue")
        if initial_source.name != name:
            raise ValueError("initial Processor source has another signal name")
        source_component = self._require_source_component(
            initial_source,
            initial_source_component,
        )
        key = id(node)
        if key in self._entries:
            raise RuntimeError("Processor node is already attached")
        node._validate_processor_source(initial_source)
        future = self._executor.submit(node._prepare_processor_application)
        future.add_done_callback(
            lambda _future, current=node: self._wake_owner_if_open(current)
        )
        self._entries[key] = _LatestOnlyProcessorEntry(
            node=node,
            source_name=name,
            prepare_future=future,
            pending_source=initial_source,
            pending_source_component=source_component,
            last_source_identity=_signal_revision_identity(initial_source),
        )

    def cancel(self, node: object) -> bool:
        """Stop accepting revisions; return whether no worker still owns it."""

        entry = self._entries.get(id(node))
        if entry is None:
            return True
        entry.cancel_requested = True
        entry.pending_source = None
        entry.pending_source_component = None
        idle = entry.prepare_future.done() and (
            entry.work_future is None or entry.work_future.done()
        )
        if idle:
            self._retire_cancelled(entry)
        return idle

    def detach(self, node: object) -> None:
        entry = self._entries.get(id(node))
        if entry is None:
            return
        entry.cancel_requested = True
        entry.pending_source = None
        entry.pending_source_component = None
        if entry.prepare_future.done() and (
            entry.work_future is None or entry.work_future.done()
        ):
            self._entries.pop(id(node), None)

    def route(
        self,
        signals: Mapping[str, ConsoleSignalValue],
        source_components: Mapping[
            str,
            _ProcessorSourceComponent,
        ],
    ) -> None:
        """Offer each attached node the exact newest accepted source revision."""

        if self._closed:
            return
        for entry in tuple(self._entries.values()):
            if entry.cancel_requested:
                continue
            source = signals.get(entry.source_name)
            if source is None:
                continue
            identity = _signal_revision_identity(source)
            if identity == entry.last_source_identity:
                continue
            try:
                entry.node._validate_processor_source(source)
                source_component = source_components.get(source.name)
                if source_component is None:
                    raise RuntimeError(
                        "Processor source has no retained producer transaction"
                    )
                source_component = self._require_source_component(
                    source,
                    source_component,
                )
            except Exception as error:
                self._retire_failed(entry, error)
                continue
            entry.last_source_identity = identity
            entry.pending_source = source
            entry.pending_source_component = source_component
            self._start_pending(entry)

    def active_bindings(self) -> tuple[tuple[object, str], ...]:
        """Return the currently admitted source edges on the owner thread."""

        return tuple(
            (entry.node, entry.source_name)
            for entry in self._entries.values()
            if not entry.cancel_requested
        )

    def source_component_for_publication(
        self,
        node: object,
        source: ConsoleSignalValue,
    ) -> _ProcessorSourceComponent:
        """Return the exact transaction retained beside the completed work."""

        entry = self._entries.get(id(node))
        if entry is None or entry.work_source is None:
            raise RuntimeError("Processor publication has no admitted work source")
        if (
            _signal_revision_identity(entry.work_source)
            != _signal_revision_identity(source)
        ):
            raise RuntimeError("Processor publication source differs from its work")
        component = entry.work_source_component
        if component is None:
            raise RuntimeError("Processor lane lost its source transaction")
        return component

    def drain(self) -> None:
        """Admit completed worker results on the TaskConsole owner thread."""

        for entry in tuple(self._entries.values()):
            if not entry.prepare_future.done():
                continue
            if entry.application is None:
                try:
                    application = entry.prepare_future.result()
                    entry.node._processor_application_ready(application)
                except Exception as error:
                    self._retire_failed(entry, error)
                    continue
                entry.application = application
            work = entry.work_future
            if work is not None and work.done():
                source = entry.work_source
                if entry.cancel_requested:
                    entry.work_future = None
                    entry.work_source = None
                    entry.work_source_component = None
                    self._retire_cancelled(entry)
                    continue
                try:
                    if source is None:
                        raise RuntimeError(
                            "Processor lane lost its source revision"
                        )
                    result = work.result()
                    entry.node._accept_processor_result(source, result)
                except Exception as error:
                    entry.work_future = None
                    entry.work_source = None
                    entry.work_source_component = None
                    self._retire_failed(entry, error)
                    continue
                entry.work_future = None
                entry.work_source = None
                entry.work_source_component = None
            if entry.cancel_requested:
                if entry.work_future is None:
                    self._retire_cancelled(entry)
                continue
            self._start_pending(entry)

    def _start_pending(self, entry: _LatestOnlyProcessorEntry) -> None:
        if (
            entry.cancel_requested
            or entry.application is None
            or entry.work_future is not None
            or entry.pending_source is None
        ):
            return
        try:
            source, entry.pending_source = entry.pending_source, None
            source_component = entry.pending_source_component
            entry.pending_source_component = None
            if source_component is None:
                raise RuntimeError("Processor lane lost its pending source transaction")
            coverage = source.coverage
            if not isinstance(coverage, MonitorCoverage):
                raise TypeError(
                    "latest-only Processor source must carry MonitorCoverage"
                )
            entry.node._processor_work_started(source)
            future = self._executor.submit(
                _evaluate_prepared_processor_application,
                entry.application,
                source,
                coverage,
            )
        except Exception as error:
            self._retire_failed(entry, error)
            return
        future.add_done_callback(
            lambda _future, current=entry.node: self._wake_owner_if_open(current)
        )
        entry.work_source = source
        entry.work_source_component = source_component
        entry.work_future = future

    @staticmethod
    def _require_source_component(
        source: ConsoleSignalValue,
        component: _ProcessorSourceComponent,
    ) -> _ProcessorSourceComponent:
        if not isinstance(component, _ProcessorSourceComponent):
            raise TypeError(
                "Processor source transaction has another component type"
            )
        selected = component.signals.get(source.name)
        if (
            selected is None
            or _signal_revision_identity(selected)
            != _signal_revision_identity(source)
        ):
            raise ValueError(
                "Processor source transaction does not contain its selected revision"
            )
        return component.through_signal(source.name)

    def _retire_failed(
        self,
        entry: _LatestOnlyProcessorEntry,
        error: Exception,
    ) -> None:
        self._entries.pop(id(entry.node), None)
        entry.node._accept_processor_failure(error)

    def _retire_cancelled(self, entry: _LatestOnlyProcessorEntry) -> None:
        self._entries.pop(id(entry.node), None)
        entry.node._accept_processor_cancelled()

    def _wake_owner_if_open(self, node: object) -> None:
        """Never enqueue an owner wake after the data plane has retired."""

        if not self._closed:
            node._request_processor_owner_wake()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for entry in tuple(self._entries.values()):
            entry.cancel_requested = True
            entry.pending_source = None
            entry.pending_source_component = None
        # Running evaluations consume only immutable inputs and publish only
        # when drain() admits them.  Once closed there is no consumer, so do
        # not block the Qt owner waiting for pure work that will be discarded.
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._entries.clear()


class ConsoleDataPlane:
    """Own live slots and coalesce their revision notifications.

    Slots arrive from the RUN seam: a node's start closure builds one and
    registers it here, so this plane never talks to the domain itself -- it
    holds what the monitor handed back and reads it atomically.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processor_lane = _LatestOnlyProcessorLane()
        self._slots: dict[int, tuple[object, object]] = {}
        self._dirty: set[int] = set()
        self._cache: dict[int, dict[str, ConsoleSignalValue]] = {}
        self._finals: dict[
            int,
            tuple[object, dict[str, ConsoleSignalValue]],
        ] = {}
        self._processors: dict[int, _ProcessorPublication] = {}
        self._panels: dict[str, dict[str, ConsoleSignalValue]] = {}
        self._failures: dict[int, str] = {}
        self._membership_changed = False
        self._closed = False
        empty = MappingProxyType({})
        self._front = ConsoleDataFront(signals=empty, failures=empty)
        self._front_source_components: Mapping[
            str,
            _ProcessorSourceComponent,
        ] = empty

    # ------------------------------------------------------------ membership
    def attach(self, node, slot) -> None:
        if slot is None:
            raise ValueError("a monitor slot is required")
        if not callable(getattr(slot, "freeze_live_outputs", None)):
            raise TypeError(
                "live slot must expose application-owned freeze_live_outputs()"
            )
        _node_instance_id(node)
        key = id(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("console data plane is closed")
            if key in self._slots:
                raise RuntimeError(
                    "console node already owns a run-scoped live route"
                )
            self._slots[key] = (node, slot)
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

    def attach_latest_only_processor(
        self,
        node: object,
        *,
        source_name: str,
        initial_source: ConsoleSignalValue,
    ) -> None:
        """Attach one processor to an explicit already-accepted source revision."""

        with self._lock:
            if self._closed:
                raise RuntimeError("console data plane is closed")
            initial_component = self._source_component_locked(
                source_name,
                initial_source,
            )
        self._processor_lane.attach(
            node,
            source_name,
            initial_source,
            initial_component,
        )

    def _source_component_locked(
        self,
        source_name: str,
        source: ConsoleSignalValue,
    ) -> _ProcessorSourceComponent:
        """Resolve one accepted source to its complete retained transaction."""

        expected = _signal_revision_identity(source)
        presented = self._front_source_components.get(source_name)
        if presented is not None:
            selected = presented.signals.get(source_name)
            if (
                selected is not None
                and _signal_revision_identity(selected) == expected
            ):
                return presented.through_signal(source_name)
        for values in self._cache.values():
            selected = values.get(source_name)
            if (
                selected is not None
                and _signal_revision_identity(selected) == expected
            ):
                return _ProcessorSourceComponent(values).through_signal(source_name)
        for publication in self._processors.values():
            component = publication.output_component()
            selected = component.signals.get(source_name)
            if (
                selected is not None
                and _signal_revision_identity(selected) == expected
            ):
                return component.through_signal(source_name)
        raise RuntimeError(
            "initial Processor source has no retained producer transaction"
        )

    def cancel_latest_only_processor(self, node: object) -> bool:
        """Stop routing new revisions to one processor."""

        idle = self._processor_lane.cancel(node)
        self.withdraw_processor(node)
        return idle

    def drain_latest_only_processors(self) -> None:
        """Admit completed shared-lane work on the TaskConsole owner thread."""

        self._processor_lane.drain()

    def withdraw_processor(self, node: object) -> None:
        """Withdraw a stopped Processor's transient publication."""

        key = id(node)
        with self._lock:
            if self._processors.pop(key, None) is not None:
                self._membership_changed = True

    def publish_final(
        self,
        node,
        projected: Mapping[str, FinalDatasetOutput],
        *,
        presentations: Mapping[str, SiteMapPresentation] | None = None,
    ) -> None:
        """Admit one successful Run's already-materialized FINAL datasets.

        ``projected`` is keyed by the catalog's bare output names.  The data
        plane qualifies them with the exact producer instance just like a live
        slot; it never invents an output that the node did not declare.
        """

        if not isinstance(projected, Mapping):
            raise TypeError("projected FINAL signals must be a mapping")
        _node_instance_id(node)
        presentations = {} if presentations is None else presentations
        if not isinstance(presentations, Mapping):
            raise TypeError("FINAL presentations must be a mapping")
        declarations = tuple(
            getattr(node, "output_declarations", ()) or ()
        )
        declared = _declared_outputs(declarations)
        output_names = tuple(projected)
        if any(
            not isinstance(name, str) or not name or name.strip() != name
            for name in output_names
        ):
            raise ValueError("FINAL output keys must be canonical text")
        actual = set(output_names)
        if not actual:
            raise ValueError("FINAL output owner must return a non-empty mapping")
        if not actual.issubset(declared):
            raise ValueError(
                "FINAL output owner published an output absent from the "
                "Workbench vocabulary: "
                f"declared={tuple(sorted(declared))}, "
                f"unknown={tuple(sorted(actual - set(declared)))}"
            )
        presentation_names = tuple(presentations)
        if any(
            not isinstance(name, str) or not name or name.strip() != name
            for name in presentation_names
        ):
            raise ValueError("FINAL presentation keys must be canonical text")
        unknown_presentations = set(presentation_names) - actual
        if unknown_presentations:
            raise ValueError(
                "FINAL presentation has no matching domain output: "
                + ", ".join(sorted(unknown_presentations))
            )
        for presentation in presentations.values():
            if not isinstance(presentation, SiteMapPresentation):
                raise TypeError(
                    "FINAL presentations must contain SiteMapPresentation values"
                )
        title = _node_display_label(node)
        handle = getattr(node, "handle", None)
        run_id_value = getattr(handle, "run_id", None)
        run_id = getattr(run_id_value, "value", run_id_value)
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(
                "a successful FINAL publication must retain its RunHandle RunId"
            )
        frozen: dict[str, ConsoleSignalValue] = {}
        for output_name, value in projected.items():
            if not isinstance(value, FinalDatasetOutput):
                raise TypeError(
                    "FINAL values must be FinalDatasetOutput"
                )
            output = value
            _require_published_declaration(
                str(output_name),
                output,
                declared,
            )
            key = node.signal_key(output.name)
            snapshot = output.snapshot
            frozen[key] = ConsoleSignalValue(
                name=key,
                source=title,
                snapshot=snapshot,
                coverage=None,
                run_id=run_id,
                epoch_id=snapshot.ref.stream_generation.value,
                join_digest=output.join_digest,
                transient=False,
                presentation=presentations.get(output.name),
            )
        key = id(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("console data plane is closed")
            self._finals[key] = (node, frozen)
            self._membership_changed = True

    def publish_processor(
        self,
        node,
        outputs: Mapping[str, LiveDatasetOutput],
        *,
        source: ConsoleSignalValue,
        presentations: Mapping[str, SiteMapPresentation] | None = None,
    ) -> None:
        """Namespace one Processor-owned typed output transaction.

        The neutral Processor owns bare names, Datasets, coverage, and join
        identity.  This composition seam validates the frozen RUN declaration,
        attaches an optional typed frontend presentation, and qualifies names.
        """

        if not isinstance(outputs, Mapping):
            raise TypeError("processor outputs must be a mapping")
        if not isinstance(source, ConsoleSignalValue):
            raise TypeError("processor source must be ConsoleSignalValue")
        _node_instance_id(node)
        presentations = {} if presentations is None else presentations
        if not isinstance(presentations, Mapping):
            raise TypeError("processor presentations must be a mapping")
        declarations = tuple(getattr(node, "output_declarations", ()) or ())
        declared = _declared_outputs(declarations)
        output_names = tuple(outputs)
        if any(
            not isinstance(name, str) or not name or name.strip() != name
            for name in output_names
        ):
            raise ValueError("processor output keys must be canonical text")
        actual = set(output_names)
        if not actual:
            raise ValueError("processor output owner must return a non-empty mapping")
        expected_outputs = set(declared)
        if actual != expected_outputs:
            raise ValueError(
                "Processor publication must cover its complete frozen output "
                f"vocabulary: missing={tuple(sorted(expected_outputs - actual))}, "
                f"unknown={tuple(sorted(actual - expected_outputs))}"
            )
        presentation_names = tuple(presentations)
        if any(
            not isinstance(name, str) or not name or name.strip() != name
            for name in presentation_names
        ):
            raise ValueError("processor presentation keys must be canonical text")
        if not set(presentation_names).issubset(actual):
            raise ValueError("processor presentation has no declared output")
        title = _node_display_label(node)
        frozen: dict[str, ConsoleSignalValue] = {}
        for output_name in outputs:
            output = outputs[output_name]
            if not isinstance(output, LiveDatasetOutput):
                raise TypeError(
                    "processor outputs must contain LiveDatasetOutput"
                )
            _require_published_declaration(output_name, output, declared)
            presentation = presentations.get(output_name)
            selected = node.signal_key(output_name)
            frozen[selected] = ConsoleSignalValue(
                name=selected,
                source=title,
                snapshot=output.snapshot,
                coverage=output.coverage,
                run_id=source.run_id,
                epoch_id=source.epoch_id,
                join_digest=output.join_digest,
                transient=True,
                presentation=presentation,
            )
        source_component = self._processor_lane.source_component_for_publication(
            node,
            source,
        )
        key = id(node)
        edge = _ProcessorEdgePublication(
            node=node,
            source_name=source.name,
            source_identity=_signal_revision_identity(source),
            outputs=frozen,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("console data plane is closed")
            self._processors[key] = _ProcessorPublication(
                edge=edge,
                source_component=source_component,
            )
            self._membership_changed = True

    def publish_panel(
        self,
        panel_id: str,
        source: ConsoleSignalValue,
        values: Mapping[str, object],
    ) -> None:
        """Route one Figure owner's bare outputs into this panel namespace."""

        from .console_records import panel_signal_key
        from zlc_frontend.figure_outputs import FigureDerivedSignal
        from zlc_neutral_atom.processing import derive_dataset_event_digest

        identity = str(panel_id).strip()
        if not identity:
            raise ValueError("panel_id must not be empty")
        if not isinstance(values, Mapping):
            raise TypeError("panel values must be a mapping")
        if not isinstance(source, ConsoleSignalValue):
            raise TypeError("panel source must be ConsoleSignalValue")
        if not values:
            raise ValueError("use withdraw_panel() to remove panel outputs")
        frozen: dict[str, ConsoleSignalValue] = {}
        for raw_name, value in values.items():
            output_name = str(raw_name)
            if not isinstance(value, FigureDerivedSignal):
                raise TypeError("panel values must contain FigureDerivedSignal")
            if value.source_ref != getattr(source.snapshot, "ref", None):
                raise ValueError("Figure output belongs to another source revision")
            name = panel_signal_key(identity, output_name)
            frozen[name] = ConsoleSignalValue(
                name=name,
                source=identity,
                snapshot=value.snapshot,
                coverage=(
                    source.coverage if value.preserve_source_coverage else None
                ),
                run_id=source.run_id,
                epoch_id=source.epoch_id,
                join_digest=derive_dataset_event_digest(
                    source.join_digest,
                    value.derivation_digest,
                ),
                transient=False,
                presentation=value.metadata,
            )
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
        self._processor_lane.detach(node)
        with self._lock:
            entry = self._slots.pop(key, None)
            self._dirty.discard(key)
            self._cache.pop(key, None)
            self._finals.pop(key, None)
            self._processors.pop(key, None)
            self._failures.pop(key, None)
            self._membership_changed = True
        if entry is not None:
            _node, slot = entry
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
            self._dirty.clear()
            self._cache.clear()
            self._finals.clear()
            self._processors.clear()
            self._panels.clear()
            self._failures.clear()
            self._membership_changed = True
        for _node, slot in entries:
            slot.close()
        self._processor_lane.close()

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

        self._processor_lane.drain()
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
            title = _node_display_label(node)
            try:
                result = self._freeze_one(node, slot, title)
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                with self._lock:
                    if self._slots.get(key) == (node, slot):
                        self._cache.pop(key, None)
                        self._failures[key] = failure
                continue
            frozen, alignment_failure = result
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
            processors = dict(self._processors)
            active_processor_sources = tuple(
                source_name
                for _node, source_name in self._processor_lane.active_bindings()
            )
            source_transactions = []
            for key, (node, _slot) in current.items():
                names = list(cached.get(key, ()))
                declared = set(_node_declared_signal_names(node))
                names.extend(
                    name
                    for name in declared
                    if name not in names
                    and (previous := self._front.signals.get(name)) is not None
                    and previous.transient
                )
                names.extend(
                    source_name
                    for source_name in active_processor_sources
                    if source_name in declared and source_name not in names
                )
                if names:
                    source_transactions.append(tuple(names))
            panels = {
                panel_id: dict(value)
                for panel_id, value in self._panels.items()
            }
            failed = dict(self._failures)
        for key, (node, _slot) in current.items():
            signals.update(cached.get(key, {}))
            failure = failed.get(key)
            if failure is not None:
                failures[_node_instance_id(node)] = failure
        for _key, (_node, values) in finals.items():
            signals.update(values)
        for publication in processors.values():
            signals.update(publication.edge.outputs)
        for _panel_id, values in panels.items():
            signals.update(values)
        # Route the newest candidate even when it is not presentable yet.  The
        # worker must see revision N in order to produce the descendant values
        # that make N coherent; presentation keeps the previous component until
        # those values arrive.
        source_components: dict[
            str,
            _ProcessorSourceComponent,
        ] = {}
        for names in source_transactions:
            root = {
                name: signals[name]
                for name in names
                if name in signals
            }
            if root:
                component = _ProcessorSourceComponent(root)
                for name in component.root_signals:
                    source_components[name] = component
        for publication in processors.values():
            component = publication.output_component()
            for name in publication.edge.outputs:
                source_components[name] = component
        self._processor_lane.route(signals, source_components)
        signals, presented_source_components = self._presentable_linked_front(
            signals,
            source_components,
            processors,
            tuple(source_transactions),
        )
        front = ConsoleDataFront(
            signals=MappingProxyType(signals),
            failures=MappingProxyType(failures),
        )
        with self._lock:
            self._front = front
            self._front_source_components = MappingProxyType(
                presented_source_components
            )
        return front

    def _presentable_linked_front(
        self,
        candidate: Mapping[str, ConsoleSignalValue],
        candidate_source_components: Mapping[
            str,
            _ProcessorSourceComponent,
        ],
        processors: Mapping[int, _ProcessorPublication],
        source_transactions: tuple[tuple[str, ...], ...],
    ) -> tuple[
        dict[str, ConsoleSignalValue],
        dict[str, _ProcessorSourceComponent],
    ]:
        """Present each already-admitted linked component as one GUI front.

        Physical causality has already been proved by the neutral processing
        owner before publication.  This method owns only presentation timing:
        if a descendant for candidate revision N has not arrived yet, the GUI
        keeps the complete prior linked front instead of displaying a mixture
        of N and N-1.  Two unrelated source components remain independent.
        """

        current = dict(candidate)
        adjacency: dict[str, set[str]] = {}
        incoherent: set[str] = set()
        selected_edges = {
            key: publication.edge
            for key, publication in processors.items()
        }
        component_candidates = list(candidate_source_components.values())

        # A completed Processor result retains the exact source transaction it
        # evaluated.  Present that completed transaction even when acquisition
        # has already advanced the raw latest candidate; the newer candidate is
        # still routed to pending work above and does not need to be discarded.
        for publication in processors.values():
            source_component = publication.source_component
            current.update(source_component.signals)
            for edge in source_component.edges:
                selected_edges[id(edge.node)] = edge
            component_candidates.append(source_component)
            component_candidates.append(publication.output_component())

        # A live slot freezes all of its declared outputs in one producer
        # transaction.  If a Processor consumes one member, staging must retain
        # its siblings too; otherwise a three-frame Camera revision could be
        # split into frame_0=N-1 and frame_1=N.
        for names in source_transactions:
            if not names:
                continue
            anchor = names[0]
            adjacency.setdefault(anchor, set()).update(names[1:])
            for name in names[1:]:
                adjacency.setdefault(name, set()).add(anchor)

        for node, source_name in self._processor_lane.active_bindings():
            output_names = _node_declared_signal_names(node)
            if not output_names:
                raise ValueError("Processor node declares no outputs")
            adjacency.setdefault(source_name, set()).update(output_names)
            for output_name in output_names:
                adjacency.setdefault(output_name, set()).add(source_name)

            source = current.get(source_name)
            edge = selected_edges.get(id(node))
            accepted_source = None if edge is None else edge.source_identity
            outputs_match = edge is not None and all(
                (presented := current.get(name)) is not None
                and _signal_revision_identity(presented)
                == _signal_revision_identity(value)
                for name, value in edge.outputs.items()
            )
            if (
                source is None
                or accepted_source != _signal_revision_identity(source)
                or not outputs_match
            ):
                incoherent.add(source_name)
                incoherent.update(output_names)

        blocked = set(incoherent)
        pending = list(incoherent)
        while pending:
            name = pending.pop()
            for neighbour in adjacency.get(name, ()):
                if neighbour not in blocked:
                    blocked.add(neighbour)
                    pending.append(neighbour)

        if blocked:
            previous = self._front.signals
            for name in blocked:
                old = previous.get(name)
                if old is None:
                    current.pop(name, None)
                else:
                    current[name] = old

        front_components: dict[str, _ProcessorSourceComponent] = {}
        for name in blocked:
            previous_component = self._front_source_components.get(name)
            if previous_component is not None and name in current:
                front_components[name] = previous_component
        for component in reversed(component_candidates):
            values = component.signals
            for name, value in values.items():
                presented = current.get(name)
                if (
                    name not in front_components
                    and presented is not None
                    and _signal_revision_identity(presented)
                    == _signal_revision_identity(value)
                ):
                    front_components[name] = component.through_signal(name)
        return current, front_components

    def _freeze_one(
        self,
        node,
        slot,
        title: str,
    ) -> tuple[dict[str, ConsoleSignalValue], str | None]:
        """Namespace one application-owned mapping onto declared outputs.

        Figure-owned Area, locked Cross, and Fit branches are derived later
        from the exact accepted panel front and never reconfigure this producer.
        """

        run_id, causation, outputs = slot.freeze_live_outputs()
        if not isinstance(run_id, str) or not run_id:
            raise TypeError("live dataset run_id must be a non-empty string")
        if not isinstance(causation, str) or not causation:
            raise TypeError(
                "live dataset causation_domain_id must be a non-empty string"
            )
        if not isinstance(outputs, Mapping) or not outputs:
            raise ValueError("live output owner must return a non-empty mapping")
        declarations = tuple(
            getattr(node, "output_declarations", ()) or ()
        )
        declared = _declared_outputs(declarations)
        bare_names = tuple(outputs)
        if any(
            not isinstance(name, str) or not name or name.strip() != name
            for name in bare_names
        ):
            raise ValueError("live output names must be canonical text")
        actual = set(bare_names)
        if not actual.issubset(declared):
            raise ValueError(
                "live output owner published an output absent from the "
                "Workbench vocabulary: "
                f"declared={tuple(sorted(declared))}, "
                f"unknown={tuple(sorted(actual - set(declared)))}"
            )
        frozen: dict[str, ConsoleSignalValue] = {}
        for output_name in bare_names:
            output = outputs[output_name]
            if not isinstance(output, LiveDatasetOutput):
                raise TypeError("live output values must be LiveDatasetOutput")
            _require_published_declaration(output_name, output, declared)
            output_snapshot = output.snapshot
            selected = node.signal_key(output_name)
            frozen[selected] = ConsoleSignalValue(
                name=selected,
                source=title,
                snapshot=output_snapshot,
                coverage=output.coverage,
                run_id=run_id,
                epoch_id=causation,
                join_digest=output.join_digest,
                transient=True,
            )
        return frozen, getattr(slot, "notification_failure", None)
