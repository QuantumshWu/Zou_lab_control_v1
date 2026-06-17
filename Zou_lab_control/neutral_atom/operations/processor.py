"""Declarative DATA-PROCESSING actions -- the "func" layer that turns signals into
derived signals on the SignalHub.

A processor publishes derived hub signals in one of two execution styles, declared
ONCE per spec (exactly one of the two builders is set):

* **reactive** (``make_node``): builds a live :class:`~.logic.Processor` node that
  CONSUMES one or more hub signals and republishes derived ones every time an input
  advances -- e.g. judging per-site occupancy from each camera ``frame``.  This is
  the textbook "func" node: it runs beside the measurement that supplies it.
* **one-shot** (``run``): runs ONCE over freshly-acquired or saved frames, produces
  a structured RESULT (per-site arrays + scalars), and hands it back as
  ``{signal_name: value}`` -- e.g. characterizing readout fidelity from a folder.
  The discrete sibling of a finite scan.

Either way the output goes to the hub as a PROCESSOR signal (measurements +
processors are the only nodes that publish to the hub; a Task's artifacts do NOT go
on the hub).  Both styles share the SAME shape: declared parameters (:class:`ParamDecl`)
in, named data out.

A processor is contributed as a FACTORY ``build(readout) -> ProcessorSpec`` and is
auto-discovered exactly like a measurement (see :mod:`processor_registry`).  It
DRIVES existing analysis / detection (e.g. ``calibration.detect`` or
``ReadoutSubsystem.characterize_from_dir``) -- it re-implements no readout /
fidelity / threshold math -- and any calibration it derives is written back through
the subsystem, never mutated in place.

Imports no concrete backend and reads no simulation ground truth: a processor's
only data source is a hub signal, ``ctx.camera.acquire`` or a saved folder, so a
virtual run exercises the identical path a real run does
(``tests/test_virtual_equals_real_contract.py`` guards this).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Reuse the ONE declarative parameter type: same kind->widget mapping and the same
# no-eval coercion the measurement form already uses (single source of truth).
from .measurement import ParamDecl


@dataclass(frozen=True)
class ProcessorContext:
    """What a processor's ``run`` is handed.

    ``readout``    the ReadoutSubsystem -- to DRIVE its characterize/calibrate
                   methods and reach the session calibration (the single owner).
    ``params``     the validated ``{key: value}`` parameter values (coerced by kind).
    ``camera``     the data SOURCE for a live grab (a ``CameraDevice``; ``None`` when
                   the action reads a saved folder instead).
    ``sequencer``  the pulse sequencer for a live grab (``None`` if not needed).
    ``stop``       a ``threading.Event`` the run polls / passes to ``camera.acquire``
                   so a long grab cancels cleanly (the cooperative-cancel path).
    """

    readout: Any
    params: dict
    camera: Any = None
    sequencer: Any = None
    stop: Any = None


@dataclass(frozen=True)
class ProcessorSpec:
    """A named processing action + its declared parameters + ONE builder.

    Set EXACTLY ONE of:

    * ``make_node(hub, *, prefix="", **param_values) -> Processor`` -- a REACTIVE
      live node (a :class:`~.logic.Processor` subclass) that consumes ``consumes``
      and republishes ``result_keys`` whenever an input advances.  The factory
      closure captures the subsystem (like a measurement's ``build``), so the
      console stays decoupled.
    * ``run(ctx) -> {signal_name: value}`` -- a ONE-SHOT action returning hub-ready
      numpy arrays / scalars (NOT a domain object); the host publishes them under
      ``result_keys``.

    ``consumes`` names the hub signals a reactive processor reads (drives the signal
    flow graph / legend); empty for a one-shot.  ``summary_keys`` names the SCALAR
    results worth showing in a panel's numeric pane (the rest are per-site arrays).

    The OPTIONAL default-view binding mirrors a measurement's (and confocal's
    ``plotter`` class attribute): ``default_kind`` is the plot kind the console
    auto-builds for this processor (e.g. ``"sites"`` for a per-site map) and
    ``default_value_key`` is the result key that view reads as its ``value``.  Leave
    ``default_kind`` EMPTY for a pure data-processing action -- the panel then just
    lists the published output names and the user wires them into a separate plot.
    ``grid_shape`` lets a consumer reshape a per-site vector into a 2-D map.
    """

    name: str
    params: tuple[ParamDecl, ...]
    result_keys: tuple[str, ...]
    run: Callable[["ProcessorContext"], dict] | None = None
    make_node: Callable[..., Any] | None = None
    consumes: tuple[str, ...] = ()
    summary_keys: tuple[str, ...] = ()
    default_kind: str = ""
    default_value_key: str = ""
    grid_shape: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Exactly one execution style -- a processor is reactive OR one-shot, never
        # both/neither (clean dispatch in the console; no ambiguous half-built spec).
        if (self.run is None) == (self.make_node is None):
            raise ValueError(
                f"ProcessorSpec {self.name!r} must set EXACTLY ONE of run (one-shot) "
                "or make_node (reactive).")

    @property
    def reactive(self) -> bool:
        """True for a live per-signal node, False for a one-shot folder/grab action."""

        return self.make_node is not None

    def param(self, key: str) -> ParamDecl:
        """Return the declaration for ``key`` (raises ``KeyError`` if absent)."""

        for decl in self.params:
            if decl.key == key:
                return decl
        raise KeyError(key)

    def defaults(self) -> dict[str, Any]:
        """The declared default value for every parameter, keyed by ``key``."""

        return {decl.key: decl.default for decl in self.params}


__all__ = ["ProcessorContext", "ProcessorSpec", "ParamDecl"]
