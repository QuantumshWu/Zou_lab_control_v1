"""MONITOR seam: change-driven latest fronts from independent producers.

Monitor seam of the composition root (``app.py``).  A run node
publishes into a ``LiveDatasetSlot``; this module freezes only slots that
reported a new revision.  Each slot is one producer transaction.  Combining
their latest immutable fronts into one presentation cycle does *not* assert
that independent producers observed the same physical shot.

Freeze-latest, not a bus.  Each changed slot materialises its own atomic
transaction exactly once; unchanged slots reuse their immutable fronts.  The resulting
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
from zlc_frontend.site_map_render import SiteMapView
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
        SiteMapView | SelectorAxisMetadata | FitParameterMetadata | None
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
            (*SiteMapView.__args__, SelectorAxisMetadata, FitParameterMetadata),
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
    """Latest immutable value of each producer; no cross-producer join claim."""

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
    return evaluate(
        source.snapshot,
        coverage,
        source_event_digest=source.join_digest,
    )


@dataclass(slots=True)
class _LatestOnlyProcessorEntry:
    node: object
    source_name: str
    prepare_future: Future
    application: object | None = None
    work_future: Future | None = None
    work_source: ConsoleSignalValue | None = None
    pending_source: ConsoleSignalValue | None = None
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
    ) -> None:
        if self._closed:
            raise RuntimeError("latest-only Processor lane is closed")
        self._require_node_contract(node)
        name = canonical_text(source_name, "processor source name")
        if not isinstance(initial_source, ConsoleSignalValue):
            raise TypeError("initial Processor source must be ConsoleSignalValue")
        if initial_source.name != name:
            raise ValueError("initial Processor source has another signal name")
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
            last_source_identity=_signal_revision_identity(initial_source),
        )

    def cancel(self, node: object) -> bool:
        """Stop accepting revisions; return whether no worker still owns it."""

        entry = self._entries.get(id(node))
        if entry is None:
            return True
        entry.cancel_requested = True
        entry.pending_source = None
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
        if entry.prepare_future.done() and (
            entry.work_future is None or entry.work_future.done()
        ):
            self._entries.pop(id(node), None)

    def route(self, signals: Mapping[str, ConsoleSignalValue]) -> None:
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
            except Exception as error:
                self._retire_failed(entry, error)
                continue
            entry.last_source_identity = identity
            entry.pending_source = source
            self._start_pending(entry)

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
                entry.work_future = None
                entry.work_source = None
                if entry.cancel_requested:
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
                    self._retire_failed(entry, error)
                    continue
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
        entry.work_future = future

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
        self._processor_lane.attach(node, source_name, initial_source)

    def cancel_latest_only_processor(self, node: object) -> bool:
        """Stop routing new revisions to one processor."""

        return self._processor_lane.cancel(node)

    def drain_latest_only_processors(self) -> None:
        """Admit completed shared-lane work on the TaskConsole owner thread."""

        self._processor_lane.drain()

    def publish_final(
        self,
        node,
        projected: Mapping[str, FinalDatasetOutput],
        *,
        presentations: Mapping[str, SiteMapView] | None = None,
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
            if not isinstance(presentation, SiteMapView.__args__):
                raise TypeError("FINAL presentations must contain SiteMapView values")
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
        run_id: str,
        epoch_id: str,
        presentations: Mapping[str, SiteMapView] | None = None,
    ) -> None:
        """Namespace one Processor-owned typed output transaction.

        The neutral Processor owns bare names, Datasets, coverage, and join
        identity.  This composition seam validates the frozen RUN declaration,
        attaches an optional typed frontend presentation, and qualifies names.
        """

        if not isinstance(outputs, Mapping):
            raise TypeError("processor outputs must be a mapping")
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
        if not actual.issubset(declared):
            raise ValueError(
                "Processor published an output absent from the Workbench vocabulary"
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
                run_id=run_id,
                epoch_id=epoch_id,
                join_digest=output.join_digest,
                transient=True,
                presentation=presentation,
            )
        key = id(node)
        with self._lock:
            if self._closed:
                raise RuntimeError("console data plane is closed")
            self._processors[key] = (node, frozen)
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
        from zlc_storage import canonical_digest

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
                join_digest=canonical_digest(
                    {
                        "owner": "zlc-workbench.task-console.figure-route",
                        "source_join_digest": source.join_digest,
                        "derivation_digest": value.derivation_digest,
                    }
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
            processors = {
                key: (node, dict(value))
                for key, (node, value) in self._processors.items()
            }
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
        for _key, (_node, values) in processors.items():
            signals.update(values)
        for _panel_id, values in panels.items():
            signals.update(values)
        self._processor_lane.route(signals)
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
