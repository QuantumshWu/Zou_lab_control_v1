"""Declarative one-shot DATA-PROCESSING actions (the discrete sibling of a scan).

A scanned measurement SWEEPS a parameter and streams a live curve; a PROCESSOR
runs ONCE over freshly-acquired or saved frames, produces a structured RESULT
(per-site arrays + scalars), and hands it back as ``{signal_name: value}`` for the
host to publish to the SignalHub.  Both are the SAME shape -- declared parameters
(:class:`ParamDecl`) in, named data out -- differing only in execution (swept vs
one-shot) and in whether they DECLARE a default plot binding.

A processor is contributed as a FACTORY ``build(readout) -> ProcessorSpec`` and is
auto-discovered exactly like a measurement (see :mod:`processor_registry`).  Its
``run(ctx)`` DRIVES the existing analysis (e.g.
``ReadoutSubsystem.characterize_from_dir``) -- it re-implements no readout /
fidelity / threshold math -- and any calibration it derives is written back through
the subsystem, never mutated in place.

Imports no concrete backend and reads no simulation ground truth: a processor's
only data source is ``ctx.camera.acquire`` or a saved folder, so a virtual run
exercises the identical path a real run does
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
    """A named one-shot processing action + its declared parameters + a run closure.

    ``run(ctx) -> {signal_name: value}`` returns hub-ready numpy arrays / scalars
    (NOT a domain object); the host publishes them under ``result_keys``.
    ``summary_keys`` names the SCALAR results worth showing in a panel's numeric
    pane (the rest are per-site arrays for a plot).

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
    run: Callable[["ProcessorContext"], dict]
    result_keys: tuple[str, ...]
    summary_keys: tuple[str, ...] = ()
    default_kind: str = ""
    default_value_key: str = ""
    grid_shape: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
