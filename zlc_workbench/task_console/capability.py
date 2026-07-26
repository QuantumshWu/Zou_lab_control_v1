"""Explicit, immutable Logic-node attachments consumed by TaskConsole.

The console is a generic host.  A concrete Logic node enters it only through
one value that pairs the node's already-projected declaration with the adapter
that creates its hosted lifecycle.  The value is assembled by the outer
composition root; it is not persisted, discovered, registered, or looked up
through a service locator.

This module deliberately knows no Camera, Calibration, Occupancy, MOT, RF or
PulseScan type.  It only resolves owner-declared Dataset/Artifact inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from zlc_neutral_atom.node_input import BoundNodeInputs
from zlc_neutral_atom.runtime.signal_source import SignalEventSource
from .input_binding import (
    ResolvedArtifactInput,
    ResolvedDatasetInput,
    ResolvedNodeInput,
    bind_resolved_node_inputs,
)

from .catalog_bridge import ConsoleNodeSpec
from .data_plane import ConsoleDataPlane, ConsoleSignalValue


@dataclass(frozen=True, slots=True)
class ConsoleNodeInputs:
    """One resolution transaction in routing and domain vocabularies.

    ``resolved`` keeps only the Workbench identities required to coordinate
    live producers.  ``bound`` is the stripped, typed value passed to the
    Logic-node owner.  Keeping both in one value prevents a concrete adapter
    from resolving the same fields twice and accidentally binding a different
    producer or artifact.
    """

    resolved: Mapping[str, ResolvedNodeInput]
    bound: BoundNodeInputs

    def __post_init__(self) -> None:
        if not isinstance(self.resolved, Mapping):
            raise TypeError("resolved must be a mapping")
        resolved = dict(self.resolved)
        if any(
            not isinstance(value, (ResolvedDatasetInput, ResolvedArtifactInput))
            for value in resolved.values()
        ):
            raise TypeError("resolved inputs contain another value type")
        if not isinstance(self.bound, BoundNodeInputs):
            raise TypeError("bound must be BoundNodeInputs")
        if tuple(resolved) != tuple(self.bound.values):
            raise ValueError("routing and domain input keys differ")
        object.__setattr__(self, "resolved", MappingProxyType(resolved))

    def only_dataset(self) -> ResolvedDatasetInput:
        """Return the sole Dataset input earned by the current Processor host."""

        values = tuple(
            value
            for value in self.resolved.values()
            if isinstance(value, ResolvedDatasetInput)
        )
        if len(values) != 1:
            raise ValueError(
                "this hosted operation requires exactly one Dataset input"
            )
        return values[0]

@dataclass(frozen=True, slots=True)
class ConsoleNodeHost:
    """Generic services available while a concrete attachment creates a node."""

    data_plane: ConsoleDataPlane
    resolve_inputs: Callable[[ConsoleNodeSpec, Mapping[str, object]], Mapping[str, ResolvedNodeInput]]
    request_owner_wake: Callable[[], None]

    def __post_init__(self) -> None:
        if not isinstance(self.data_plane, ConsoleDataPlane):
            raise TypeError("data_plane must be ConsoleDataPlane")
        for name in (
            "resolve_inputs",
            "request_owner_wake",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

    def bind_inputs(
        self,
        spec: ConsoleNodeSpec,
        values: Mapping[str, object],
        *,
        resolve_artifact_reference: Callable[[ResolvedArtifactInput], object]
        | None = None,
    ) -> ConsoleNodeInputs:
        """Resolve once, then strip GUI/runtime identity for the domain owner."""

        resolved = dict(self.resolve_inputs(spec, values))
        bound = bind_resolved_node_inputs(
            resolved,
            resolve_artifact_reference=resolve_artifact_reference,
        )
        return ConsoleNodeInputs(resolved, bound)

    def current_value(self, binding: ResolvedDatasetInput) -> ConsoleSignalValue:
        """Return the currently admitted immutable value of one selected input."""

        if not isinstance(binding, ResolvedDatasetInput):
            raise TypeError("binding must be ResolvedDatasetInput")
        value = self.data_plane.freeze().value(binding.selection.signal_key)
        if not isinstance(value, ConsoleSignalValue):
            raise RuntimeError(
                "the selected running Dataset producer has not published a value"
            )
        return value


@runtime_checkable
class ConsoleSignalEventSourceProvider(Protocol):
    """Stable Workbench seam for a running row's actual event source.

    The provider itself never pretends to own optional association methods.
    Callers inspect the returned neutral source against the exact capability
    protocol they require.  The row's Python type therefore stays stable across
    prepare, start, and terminal transitions.
    """

    def signal_event_source(self) -> SignalEventSource:
        """Return the currently running typed source or raise explicitly."""


ConsoleNodeFactory = Callable[
    [
        ConsoleNodeHost,
        ConsoleNodeSpec,
        Mapping[str, object],
        str,
        str,
    ],
    object,
]


@dataclass(frozen=True, slots=True)
class ConsoleCapabilityAttachment:
    """One explicit Definition -> presenter/lifecycle connection.

    ``spec`` contains only UI projection of owner declarations.  ``create_node``
    is an ephemeral composition adapter; it is never stored in the Definition
    or serialized.  Duplicate keys are rejected when the closed tuple is
    admitted by :class:`ConsoleCatalogView`.
    """

    spec: ConsoleNodeSpec
    create_node: ConsoleNodeFactory

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ConsoleNodeSpec):
            raise TypeError("spec must be ConsoleNodeSpec")
        if not callable(self.create_node):
            raise TypeError("create_node must be callable")

    @property
    def key(self):
        return self.spec.key


__all__ = [
    "ConsoleCapabilityAttachment",
    "ConsoleNodeFactory",
    "ConsoleNodeHost",
    "ConsoleNodeInputs",
    "ConsoleSignalEventSourceProvider",
]
