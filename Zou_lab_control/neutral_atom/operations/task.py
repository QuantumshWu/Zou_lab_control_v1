"""Declarative TASK catalog entry (the orchestration layer's catalog source).

A task is the orchestration tier of the five layers (device / measurement /
processor / task / plot): a one-shot flow that may drive several devices /
measurements, save frames, derive an artifact (a calibration, an npz), and stream
MID-RUN output to its own dedicated panel -- confocal's "task" idea.

A task is contributed as a FACTORY ``build(readout) -> TaskSpec`` and auto-discovered
exactly like a measurement / processor (see :mod:`task_registry`).  The spec is a
thin declarative record the (decoupled) GUI consumes: it names the task and carries
a ``build(hub) -> Task`` closure (capturing the readout subsystem, so the console
never holds the session) plus which MID-RUN signal its dedicated panel shows.  The
task's editable parameters + Run come from the :class:`~.logic.Task` itself (its
``acquisition_parameters`` / one-shot ``start``), so there is ONE param mechanism --
no parallel form engine.

Imports no concrete backend and never touches the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

# Reuse the ONE declarative parameter type (same kind->widget mapping + no-eval
# coercion the measurement / processor forms use): a task declares its tunable
# parameters as ParamDecls so the GUI auto-generates the SAME form, and build()
# receives the validated values.
from .measurement import ParamDecl
from ._spec import REQUIRED, CatalogSpec


@dataclass(frozen=True)
class TaskSpec(CatalogSpec):
    """A named orchestration task + its declared parameters + how to build it.

    ``build(hub, *, prefix=..., **param_values) -> Task`` returns an UNRUN
    :class:`~.logic.Task` over the session (the closure captures the readout
    subsystem) configured with the chosen parameter values.  ``params`` declares
    the task's tunable parameters as :class:`ParamDecl`s, so a GUI auto-generates
    the SAME validated form a measurement/processor uses (single param mechanism).
    ``mid_run_key`` is the key into the task's mid-run output BUFFER
    (:class:`~.logic.TaskOutput`) that its dedicated panel shows while it runs (e.g.
    ``frame`` -> the template frame), ``default_kind`` is that panel's plot kind.  A
    task's output is a per-task buffer, NOT the hub (the hub carries only measurement +
    processor outputs), so ``prefix`` merely identifies the task node -- it namespaces
    no hub signal.  ``prefix`` is REQUIRED (no default) so a new task cannot silently
    inherit another task's namespace and collide at discovery."""

    build: Callable[..., Any] = REQUIRED   # build(hub, *, prefix=..., **param_values) -> Task
    prefix: str = REQUIRED
    mid_run_key: str = "frame"
    default_kind: str = "2d"

    collision_advice: ClassVar[str] = (
        "give each task a unique prefix so their signals do not collide.")

    def __post_init__(self) -> None:
        super().__post_init__()
        if not callable(self.build):
            raise TypeError(f"TaskSpec {self.name!r}: build must be callable.")
        if not self.prefix:
            raise ValueError(f"TaskSpec {self.name!r}: prefix must be a non-empty namespace token.")

    def collision_key(self) -> tuple:
        # Two tasks sharing a node prefix would collide at discovery.
        return (self.prefix,)


__all__ = ["TaskSpec", "ParamDecl"]
